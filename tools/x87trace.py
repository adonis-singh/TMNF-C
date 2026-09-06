#!/usr/bin/env python3
"""Symbolically execute the x87 stack in an exported Ghidra .asm file.

The output is an evaluation recipe for every x87 memory store. Arithmetic is
wrapped in r24()/r53() to show the configured x87 precision-control rounding;
f32()/f64() mark memory-store rounding. Integer instructions are interpreted
only far enough to normalize stack-local addresses and copied float bit
patterns.
"""

from __future__ import annotations

import argparse
import copy
import glob
import os
import re
import struct
import sys
from dataclasses import dataclass, field


LINE_RE = re.compile(r"^0x([0-9A-Fa-f]+)\s+([A-Z0-9.]+)(?:\s+(.*?))?(?:\s+;.*)?$")
MEM_RE = re.compile(
    r"^(?:(float|double|dword|qword|word)\s+ptr\s+)?\[(.+)\]$", re.IGNORECASE
)
ST_RE = re.compile(r"^ST(?:\(?([0-7])\)?|([0-7]))?$", re.IGNORECASE)
REGS = {"EAX", "EBX", "ECX", "EDX", "ESI", "EDI", "EBP", "ESP"}

# Set by --opaque-stores: name stored values by their store site instead of
# re-inlining the expression on every reload.
OPAQUE_STORES = False

# CRT intrinsics that consume ST0 and return their result in ST0.
CI_UNARY = {
    0x009C1EE0: "cos",
    0x009C2010: "sin",
}

# __thiscall float(float) getters on CSceneVehicleCarTuning. Each pops its
# one stack argument and returns the curve value in ST0.
FLOAT_RETURN_CALLS = {
    0x007F3BE0: "AccelFromSpeed",
    0x007F3C30: "RolloverLateralFromSpeed",
    0x007F3C70: "MaxSideFrictionFromSpeed",
    0x007F3CB0: "LateralContactSlowDownFromSpeed",
    0x007F3D00: "SteerSlowDownFromSpeed",
    0x007F3D50: "RolloverLateralCoefFromAngle",
    0x007F3D80: "SteerDriveTorqueFromSpeed",
    0x007F3DC0: "M4SteerRadiusFromSpeed",
    0x007F3E00: "M4MaxFrictionForceFromSpeed",
    0x007F3E40: "M5AccelFromSpeed",
    0x007F3E80: "M5SlippingAccelFromSpeed",
    0x007F3ED0: "M5SteerSlowDownFromSpeed",
    0x007F3F10: "M5LateralContactSlowDownFromSpeed",
    0x007F3F40: "WaterFrictionFromSpeed",
    0x007F3F80: "M6ModulationFromDamperAbsorbVal",
    0x007F3FF0: "M6RearGearAccelFromSpeed",
    0x007F4030: "M6BurnoutRadiusFromSpeed",
    0x007F4070: "M6LateralSpeedFromBurnoutRadius",
    0x007F40B0: "M6BurnoutRolloverFromSpeed",
    0x007F40F0: "M6DonutRolloverFromSpeed",
    0x007F4130: "M6RolloverLateralFromSpeedRatio",
}


@dataclass(frozen=True)
class Word:
    """A 32-bit integer value, optionally copied verbatim from memory."""

    base: str = ""
    offset: int = 0
    raw_source: str | None = None

    def pointer(self) -> str:
        base = f"u32[{self.raw_source}]" if self.raw_source is not None else self.base
        return add_offset(base, self.offset)

    def float_bits(self) -> str:
        if self.raw_source is not None and self.offset == 0:
            return f"f32[{self.raw_source}]"
        return f"bits_f32({self.pointer()})"


@dataclass(frozen=True)
class Expr:
    text: str


@dataclass
class Instruction:
    addr: int
    op: str
    args: list[str]
    text: str
    machine: bytes = b""


@dataclass
class State:
    stack: list[Expr] = field(default_factory=list)
    regs: dict[str, Word] = field(default_factory=dict)
    mem: dict[str, Expr | Word] = field(default_factory=dict)

    def clone(self) -> "State":
        return copy.deepcopy(self)

    def signature(self) -> tuple:
        return (
            tuple(x.text for x in self.stack),
            tuple(sorted((k, v) for k, v in self.regs.items())),
            tuple(
                sorted(
                    (k, ("e", v.text) if isinstance(v, Expr) else ("w", v))
                    for k, v in self.mem.items()
                )
            ),
        )


def add_offset(base: str, offset: int) -> str:
    if offset == 0:
        return base
    if base == "0":
        return ("-" if offset < 0 else "") + f"0x{abs(offset):x}"
    sign = "+" if offset > 0 else "-"
    return f"{base}{sign}0x{abs(offset):x}"


def split_args(text: str | None) -> list[str]:
    if not text:
        return []
    return [part.strip() for part in text.split(",")]


def parse_asm(path: str) -> list[Instruction]:
    result = []
    with open(path, encoding="utf-8") as src:
        for raw in src:
            line = raw.strip()
            match = LINE_RE.match(line)
            if not match:
                continue
            result.append(
                Instruction(
                    int(match.group(1), 16),
                    match.group(2).upper(),
                    split_args(match.group(3)),
                    line,
                )
            )
    if not result:
        raise ValueError(f"no instructions parsed from {path}")
    return result


def parse_int(text: str) -> int | None:
    try:
        return int(text, 0)
    except ValueError:
        return None


def st_index(text: str) -> int | None:
    match = ST_RE.match(text.replace(" ", ""))
    if not match:
        return None
    return int(match.group(1) or match.group(2) or 0)


def effective_address(text: str, state: State) -> str:
    """Canonicalize simple [REG +/- constant] exported operands."""

    compact = text.replace(" ", "")
    terms = re.findall(r"[+-]?[^+-]+", compact)
    base_parts: list[str] = []
    offset = 0
    for term in terms:
        sign = -1 if term.startswith("-") else 1
        atom = term[1:] if term[:1] in "+-" else term
        upper = atom.upper()
        if upper in REGS:
            word = state.regs[upper]
            base_parts.append(word.pointer())
            continue
        value = parse_int(atom)
        if value is not None:
            offset += sign * value
            continue
        base_parts.append(("-" if sign < 0 else "") + atom)
    base = "+".join(base_parts) if base_parts else "0"
    return add_offset(base, offset)


def memory_operand(text: str, state: State) -> tuple[str | None, str] | None:
    match = MEM_RE.match(text)
    if not match:
        return None
    return (match.group(1).lower() if match.group(1) else None,
            effective_address(match.group(2), state))


def load_float(text: str, state: State) -> Expr:
    parsed = memory_operand(text, state)
    if parsed is None:
        raise ValueError(f"expected memory operand, got {text}")
    width, address = parsed
    saved = state.mem.get(address)
    if isinstance(saved, Expr):
        return saved
    if isinstance(saved, Word):
        return Expr(saved.float_bits())
    if width in ("double", "qword"):
        return Expr(f"f64[{address}]")
    return Expr(f"f32[{address}]")


def load_integer(text: str, state: State) -> Word:
    parsed = memory_operand(text, state)
    if parsed is None:
        value = parse_int(text)
        return Word(base=text if value is None else hex(value))
    _, address = parsed
    saved = state.mem.get(address)
    if isinstance(saved, Word):
        return saved
    return Word(raw_source=address)


def rounded(expr: str, pc: int) -> Expr:
    return Expr(f"r{pc}({expr})")


def binary(op: str, left: Expr, right: Expr, pc: int) -> Expr:
    return rounded(f"{left.text} {op} {right.text}", pc)


def result_width(width: str | None) -> str:
    if width in ("double", "qword"):
        return "f64"
    if width == "word":
        return "i16"
    return "f32"


def store_memory(
    inst: Instruction,
    operand: str,
    value: Expr | Word,
    state: State,
    outputs: set[tuple[int, str, str]],
    integer: bool = False,
) -> None:
    parsed = memory_operand(operand, state)
    if parsed is None:
        raise ValueError(f"expected memory destination, got {operand}")
    width, address = parsed
    if isinstance(value, Word):
        state.mem[address] = value
        return
    if integer:
        marker = {"word": "i16", "qword": "i64"}.get(width, "i32")
    else:
        marker = result_width(width)
    stored = Expr(f"{marker}({value.text})")
    if OPAQUE_STORES:
        # Later loads see a short handle naming the store site instead of the
        # full expression, so long functions stay readable. f32 stores are
        # exact PC=24 values, so the handle carries the same rounding.
        state.mem[address] = Expr(f"{marker}@{inst.addr:08X}[{address}]")
    else:
        state.mem[address] = stored
    outputs.add((inst.addr, address, stored.text))


def arithmetic(inst: Instruction, state: State, pc: int) -> None:
    reverse = inst.op.startswith("FSUBR") or inst.op.startswith("FDIVR")
    symbol = {
        "FADD": "+",
        "FSUB": "-",
        "FSUBR": "-",
        "FMUL": "*",
        "FDIV": "/",
        "FDIVR": "/",
    }[inst.op.rstrip("P")]
    popping = inst.op.endswith("P")

    if popping:
        if len(state.stack) < 2:
            raise IndexError("x87 stack underflow")
        dst = st_index(inst.args[0]) if inst.args else 1
        if dst is None or dst >= len(state.stack):
            raise IndexError("invalid x87 destination")
        top, other = state.stack[0], state.stack[dst]
        left, right = (top, other) if reverse else (other, top)
        state.stack[dst] = binary(symbol, left, right, pc)
        state.stack.pop(0)
        return

    if not state.stack:
        raise IndexError("x87 stack underflow")
    if not inst.args:
        raise ValueError(f"missing operand for {inst.op}")
    if len(inst.args) == 2:
        dst = st_index(inst.args[0])
        src = st_index(inst.args[1])
        if dst is None or src is None:
            raise ValueError(f"unsupported register form: {inst.text}")
        left, right = state.stack[dst], state.stack[src]
        if reverse:
            left, right = right, left
        state.stack[dst] = binary(symbol, left, right, pc)
        return
    src_index = st_index(inst.args[0])
    if (
        src_index is not None
        and src_index != 0
        and inst.machine
        and inst.machine[0] == 0xDC
    ):
        left, right = state.stack[src_index], state.stack[0]
        if reverse:
            left, right = right, left
        state.stack[src_index] = binary(symbol, left, right, pc)
        return
    other = (
        state.stack[src_index]
        if src_index is not None
        else load_float(inst.args[0], state)
    )
    left, right = state.stack[0], other
    if reverse:
        left, right = right, left
    state.stack[0] = binary(symbol, left, right, pc)


def call_pop_bytes(asm_dir: str) -> dict[int, int]:
    result: dict[int, int] = {}
    for path in glob.glob(os.path.join(asm_dir, "*.asm")):
        name = os.path.basename(path)
        try:
            va = int(name.split("_", 1)[0], 16)
        except ValueError:
            continue
        pops: set[int] = set()
        for inst in parse_asm(path):
            if inst.op == "RET":
                pops.add(parse_int(inst.args[0]) if inst.args else 0)
        if len(pops) == 1:
            result[va] = pops.pop()
    return result


def read_pe_virtual_bytes(path: str) -> tuple[int, list[tuple[int, int, bytes]]]:
    """Return PE image base and (RVA start, virtual size, raw bytes) sections."""

    with open(path, "rb") as src:
        image = src.read()
    pe = struct.unpack_from("<I", image, 0x3C)[0]
    if image[pe:pe + 4] != b"PE\0\0":
        raise ValueError(f"{path} is not a PE image")
    section_count = struct.unpack_from("<H", image, pe + 6)[0]
    optional_size = struct.unpack_from("<H", image, pe + 20)[0]
    optional = pe + 24
    if struct.unpack_from("<H", image, optional)[0] != 0x10B:
        raise ValueError("only PE32 images are supported")
    image_base = struct.unpack_from("<I", image, optional + 28)[0]
    section_table = optional + optional_size
    sections = []
    for index in range(section_count):
        header = section_table + index * 40
        virtual_size, rva, raw_size, raw_offset = struct.unpack_from(
            "<IIII", image, header + 8
        )
        sections.append((rva, virtual_size, image[raw_offset:raw_offset + raw_size]))
    return image_base, sections


def attach_machine_bytes(insts: list[Instruction], image_path: str) -> None:
    image_base, sections = read_pe_virtual_bytes(image_path)
    for inst in insts:
        rva = inst.addr - image_base
        for start, virtual_size, raw in sections:
            if start <= rva < start + virtual_size:
                offset = rva - start
                inst.machine = raw[offset:offset + 2]
                break


def simulate_instruction(
    inst: Instruction,
    state: State,
    pc: int,
    outputs: set[tuple[int, str, str]],
    callee_pops: dict[int, int],
) -> None:
    op = inst.op
    if op == "FLD":
        index = st_index(inst.args[0])
        state.stack.insert(
            0,
            state.stack[index] if index is not None else load_float(inst.args[0], state),
        )
    elif op == "FLD1":
        state.stack.insert(0, Expr("1.0"))
    elif op == "FLDZ":
        state.stack.insert(0, Expr("0.0"))
    elif op == "FILD":
        width, address = memory_operand(inst.args[0], state)  # type: ignore[misc]
        marker = {"word": "i16", "qword": "i64"}.get(width, "i32")
        state.stack.insert(0, Expr(f"{marker}[{address}]"))
    elif op in {
        "FADD", "FADDP", "FSUB", "FSUBP", "FSUBR", "FSUBRP",
        "FMUL", "FMULP", "FDIV", "FDIVP", "FDIVR", "FDIVRP",
    }:
        arithmetic(inst, state, pc)
    elif op == "FXCH":
        index = st_index(inst.args[0]) if inst.args else 1
        if index is None:
            raise ValueError(f"bad FXCH operand: {inst.text}")
        state.stack[0], state.stack[index] = state.stack[index], state.stack[0]
    elif op == "FCHS":
        state.stack[0] = Expr(f"-{state.stack[0].text}")
    elif op == "FABS":
        state.stack[0] = Expr(f"abs({state.stack[0].text})")
    elif op == "FSQRT":
        state.stack[0] = Expr(f"sqrt{pc}({state.stack[0].text})")
    elif op in ("FST", "FSTP"):
        index = st_index(inst.args[0])
        if index is not None:
            state.stack[index] = state.stack[0]
        else:
            store_memory(inst, inst.args[0], state.stack[0], state, outputs)
        if op == "FSTP":
            state.stack.pop(0)
    elif op == "FISTP":
        value = Expr(f"fistp({state.stack[0].text})")
        store_memory(inst, inst.args[0], value, state, outputs, integer=True)
        state.stack.pop(0)
    elif op in ("FCOMPP", "FUCOMPP"):
        del state.stack[:2]
    elif op in ("FCOMP", "FUCOMP"):
        state.stack.pop(0)
    elif op in ("FCOM", "FUCOM", "FNSTSW", "FWAIT"):
        pass
    elif op == "MOV" and len(inst.args) == 2:
        dst, src = inst.args
        if dst.upper() in REGS:
            state.regs[dst.upper()] = (
                state.regs[src.upper()] if src.upper() in REGS else load_integer(src, state)
            )
        elif memory_operand(dst, state) is not None:
            value = (
                state.regs[src.upper()]
                if src.upper() in REGS
                else load_integer(src, state)
            )
            _, address = memory_operand(dst, state)  # type: ignore[misc]
            state.mem[address] = value
    elif op == "LEA" and len(inst.args) == 2 and inst.args[0].upper() in REGS:
        parsed = memory_operand(inst.args[1], state)
        if parsed is not None:
            state.regs[inst.args[0].upper()] = Word(base=parsed[1])
    elif op in ("ADD", "SUB") and len(inst.args) == 2 and inst.args[0].upper() in REGS:
        amount = parse_int(inst.args[1])
        if amount is not None:
            reg = inst.args[0].upper()
            old = state.regs[reg]
            delta = amount if op == "ADD" else -amount
            state.regs[reg] = Word(old.base, old.offset + delta, old.raw_source)
    elif op == "PUSH":
        esp = state.regs["ESP"]
        state.regs["ESP"] = Word(esp.base, esp.offset - 4, esp.raw_source)
        state.mem[state.regs["ESP"].pointer()] = (
            state.regs[inst.args[0].upper()]
            if inst.args and inst.args[0].upper() in REGS
            else load_integer(inst.args[0], state)
        )
    elif op == "POP":
        esp = state.regs["ESP"]
        if inst.args and inst.args[0].upper() in REGS:
            state.regs[inst.args[0].upper()] = load_integer(
                f"dword ptr [{esp.pointer()}]", state
            )
        state.regs["ESP"] = Word(esp.base, esp.offset + 4, esp.raw_source)
    elif op == "CALL":
        target = parse_int(inst.args[0]) if inst.args else None
        if target == 0x009C1B40:
            if not state.stack:
                raise IndexError("__CIsqrt called with empty x87 stack")
            state.stack[0] = Expr(f"sqrt{pc}({state.stack[0].text})")
        elif target in CI_UNARY:
            if not state.stack:
                raise IndexError(f"{CI_UNARY[target]} called with empty x87 stack")
            state.stack[0] = Expr(f"{CI_UNARY[target]}{pc}({state.stack[0].text})")
        elif target == 0x009C210A:
            if len(state.stack) < 2:
                raise IndexError("__CIatan2 called with fewer than two x87 values")
            # MSVC loads y then x, so ST0 is the second C argument.
            x, y = state.stack[0], state.stack[1]
            del state.stack[:2]
            state.stack.insert(0, Expr(f"atan2{pc}({y.text}, {x.text})"))
        elif target in FLOAT_RETURN_CALLS:
            # __thiscall curve getter: the float argument sits at [ESP] when
            # the CALL executes and comes back in ST0.
            argument = load_float(f"dword ptr [{state.regs['ESP'].pointer()}]", state)
            state.stack.insert(
                0, Expr(f"{FLOAT_RETURN_CALLS[target]}({argument.text})"))
        pop = callee_pops.get(target, 0) if target is not None else 0
        if pop:
            esp = state.regs["ESP"]
            state.regs["ESP"] = Word(esp.base, esp.offset + pop, esp.raw_source)


def successors(insts: list[Instruction], index: int, by_addr: dict[int, int]) -> list[int]:
    inst = insts[index]
    if inst.op == "RET":
        return []
    if inst.op == "JMP":
        target = parse_int(inst.args[0])
        return [by_addr[target]] if target in by_addr else []
    if inst.op.startswith("J") and inst.op != "JMP":
        result = []
        target = parse_int(inst.args[0]) if inst.args else None
        if target in by_addr:
            result.append(by_addr[target])
        if index + 1 < len(insts):
            result.append(index + 1)
        return result
    return [index + 1] if index + 1 < len(insts) else []


def run(
    path: str, pc: int, image_path: str | None
) -> tuple[set[tuple[int, str, str]], list[str]]:
    insts = parse_asm(path)
    if image_path is not None:
        attach_machine_bytes(insts, image_path)
    by_addr = {inst.addr: i for i, inst in enumerate(insts)}
    asm_dir = os.path.dirname(path)
    callee_pops = call_pop_bytes(asm_dir)
    initial = State(
        regs={
            "EAX": Word("eax0"),
            "EBX": Word("ebx0"),
            "ECX": Word("this"),
            "EDX": Word("edx0"),
            "ESI": Word("esi0"),
            "EDI": Word("edi0"),
            "EBP": Word("ebp0"),
            "ESP": Word("esp0"),
        }
    )
    work = [(0, initial)]
    seen: set[tuple[int, tuple]] = set()
    outputs: set[tuple[int, str, str]] = set()
    warnings: list[str] = []
    if image_path is None:
        for inst in insts:
            if (
                inst.op in {"FADD", "FSUB", "FSUBR", "FMUL", "FDIV", "FDIVR"}
                and len(inst.args) == 1
                and (st_index(inst.args[0]) or 0) != 0
            ):
                warnings.append(
                    f"0x{inst.addr:08X}: ambiguous x87 register destination; "
                    "pass --image for exact decoding"
                )
    steps = 0
    while work:
        index, state = work.pop()
        key = (index, state.signature())
        if key in seen:
            continue
        seen.add(key)
        steps += 1
        if steps > 100000:
            raise RuntimeError("symbolic execution exceeded 100000 states")
        inst = insts[index]
        try:
            simulate_instruction(inst, state, pc, outputs, callee_pops)
        except (IndexError, KeyError, TypeError, ValueError) as error:
            warnings.append(f"0x{inst.addr:08X}: {error}: {inst.text}")
            continue
        for next_index in successors(insts, index, by_addr):
            work.append((next_index, state.clone()))
    return outputs, sorted(set(warnings))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Emit symbolic x87 evaluation recipes from Ghidra .asm"
    )
    parser.add_argument("asm", help="exported function .asm file")
    parser.add_argument(
        "--pc",
        type=int,
        choices=(24, 53),
        default=24,
        help="x87 precision-control significand bits (default: 24)",
    )
    parser.add_argument(
        "--image",
        help="original PE32 image, used to disambiguate D8/DC register destinations",
    )
    parser.add_argument(
        "--opaque-stores",
        action="store_true",
        help="refer to stored values by store site (f32@VA[addr]) instead of "
        "inlining their expression on reload",
    )
    args = parser.parse_args()
    global OPAQUE_STORES
    OPAQUE_STORES = args.opaque_stores
    outputs, warnings = run(args.asm, args.pc, args.image)
    for addr, destination, expression in sorted(outputs):
        print(f"0x{addr:08X}  [{destination}] = {expression}")
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 1 if warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
