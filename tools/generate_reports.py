#!/usr/bin/env python3

import argparse
import csv
import shutil
from collections import Counter, defaultdict
from pathlib import Path


PHYSICS_STEP_VA = "0x00549C90"
KEY_VAS = (
    "0x00549C90",
    "0x005497C0",
    "0x004FE500",
    "0x004FE7E0",
)
INTEGER_FIELDS = (
    "size_bytes",
    "depth",
    "direct_callees",
    "indirect_call_sites",
    "instruction_count",
    "x87_count",
    "sse_scalar_count",
    "sse_packed_count",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def number_rows(rows: list[dict[str, str]]) -> list[dict[str, str | int]]:
    numbered: list[dict[str, str | int]] = []
    for row in rows:
        item: dict[str, str | int] = dict(row)
        for field in INTEGER_FIELDS:
            item[field] = int(item[field])
        numbered.append(item)
    return numbered


def percent(part: int, total: int) -> str:
    return f"{part * 100.0 / total:.2f}%"


def fp_totals(rows: list[dict[str, str | int]]) -> tuple[int, int, int, int]:
    x87 = sum(int(row["x87_count"]) for row in rows)
    scalar = sum(int(row["sse_scalar_count"]) for row in rows)
    packed = sum(int(row["sse_packed_count"]) for row in rows)
    return x87, scalar, packed, x87 + scalar + packed


def fp_verdict(x87: int, scalar: int, packed: int) -> str:
    active = sum(count > 0 for count in (x87, scalar, packed))
    if active > 1:
        return "mixed"
    if x87:
        return "x87"
    if scalar or packed:
        return "SSE"
    return "no floating-point instructions"


def format_fp_counts(rows: list[dict[str, str | int]]) -> str:
    x87, scalar, packed, total = fp_totals(rows)
    return (
        f"{total:,} classified FP instructions: {x87:,} x87 "
        f"({percent(x87, total)}), {scalar:,} SSE scalar "
        f"({percent(scalar, total)}), and {packed:,} SSE packed "
        f"({percent(packed, total)})."
    )


def write_fp_report(
    output: Path,
    rows: list[dict[str, str | int]],
    controls: list[dict[str, str]],
) -> None:
    physics_rows = [
        row for row in rows if PHYSICS_STEP_VA in str(row["roots"]).split(";")
    ]
    physics_counts = fp_totals(physics_rows)
    union_counts = fp_totals(rows)
    total_instructions = sum(int(row["instruction_count"]) for row in rows)
    indirect_calls = sum(int(row["indirect_call_sites"]) for row in rows)
    max_depth = max(int(row["depth"]) for row in rows)
    largest = sorted(rows, key=lambda row: int(row["size_bytes"]), reverse=True)[:10]
    indirect_top = sorted(
        (row for row in rows if int(row["indirect_call_sites"]) > 0),
        key=lambda row: int(row["indirect_call_sites"]),
        reverse=True,
    )[:10]
    tree_controls = [site for site in controls if site["in_call_tree"] == "true"]
    control_counts = Counter(site["mnemonic"] for site in controls)

    lines = [
        "# TMNF physics floating-point report",
        "",
        "## Verdict",
        "",
        (
            f"The PhysicsStep2-rooted static call graph is "
            f"**{fp_verdict(*physics_counts[:3])}**. "
            f"{format_fp_counts(physics_rows)}"
        ),
        "",
        (
            f"The union of all four requested roots is "
            f"**{fp_verdict(*union_counts[:3])}**. {format_fp_counts(rows)}"
        ),
        "",
        (
            "A bit-exact C port must preserve x87 evaluation order, its extended "
            "exponent range, the active precision-control setting, and every "
            "explicit float or double spill. No SSE floating-point operation was "
            "found in the requested graph."
        ),
        "",
        "## Call-tree statistics",
        "",
        f"- Functions: {len(rows):,}",
        f"- Maximum minimum depth: {max_depth}",
        f"- Instructions: {total_instructions:,}",
        f"- Unresolved indirect call sites: {indirect_calls:,}",
        "",
        "### Ten largest functions",
        "",
    ]
    for row in largest:
        lines.append(
            f"- `{row['va']}` {row['demangled_name']}: "
            f"{int(row['size_bytes']):,} bytes, "
            f"{int(row['instruction_count']):,} instructions"
        )

    lines.extend(["", "### Most indirect call sites", ""])
    if indirect_top:
        for row in indirect_top:
            lines.append(
                f"- `{row['va']}` {row['demangled_name']}: "
                f"{int(row['indirect_call_sites']):,}"
            )
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Floating-point control state",
            "",
            (
                "Whole-binary control instruction counts: "
                + ", ".join(
                    f"{mnemonic}={control_counts.get(mnemonic, 0)}"
                    for mnemonic in ("FLDCW", "FNSTCW", "LDMXCSR", "STMXCSR")
                )
                + "."
            ),
            "",
            (
                "CRT initialization requests `_PC_53` (`0x00010000`) under "
                "`_MCW_PC` (`0x00030000`) in `__setdefaultprecision` at "
                "`0x00404078`. `__fpmath` calls it at `0x00402AD4` when the CRT "
                "floating-point initialization flag is nonzero. This selects a "
                "53-bit x87 significand precision while retaining the x87 extended "
                "exponent range."
            ),
            "",
            (
                "PhysicsStep2 saves the x87 control word at `0x00549E61`, ORs "
                "`0x0C00` into its rounding-control field, loads it at "
                "`0x00549E89`, executes `FISTP`, and restores the saved word at "
                "`0x00549E9F`. Rounding-control value 3 is round toward zero. The "
                "sequence does not change the precision-control bits."
            ),
            "",
            (
                "No LDMXCSR or STMXCSR instruction is in the requested call tree. "
                "The binary's two LDMXCSR instructions are confined to the CRT "
                "`___set_fpsr_sse2` helper."
            ),
            "",
            "### Sites inside the requested call tree",
            "",
        ]
    )
    if tree_controls:
        for site in tree_controls:
            lines.append(
                f"- `{site['va']}` `{site['mnemonic']}` in "
                f"{site['function_name']}: `{site['instruction']}`"
            )
    else:
        lines.append("- None")

    lines.extend(["", "### All sites in the binary", ""])
    for site in controls:
        function_name = site["function_name"] or "<no containing function>"
        lines.append(
            f"- `{site['va']}` `{site['mnemonic']}` in "
            f"{function_name}: `{site['instruction']}`"
        )

    lines.extend(
        [
            "",
            (
                "Port strategy: initialize x87 to 53-bit precision, keep round-to-"
                "nearest as the ambient mode, reproduce the local round-toward-zero "
                "conversion sequence, and preserve compiler spill points. MXCSR "
                "does not affect the classified physics arithmetic because the "
                "graph contains no SSE floating-point instructions."
            ),
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def write_inventory_summary(
    output: Path,
    rows: list[dict[str, str | int]],
    edges: list[dict[str, str]],
) -> None:
    module_counts = Counter(str(row["module"]) for row in rows)
    library_counts = Counter(
        module.split(":", 1)[0] if ":" in module else "<root>"
        for module in module_counts
        for _ in range(module_counts[module])
    )
    edges_by_caller: dict[str, list[dict[str, str]]] = defaultdict(list)
    for edge in edges:
        edges_by_caller[edge["caller_va"]].append(edge)

    max_depth = max(int(row["depth"]) for row in rows)
    instruction_count = sum(int(row["instruction_count"]) for row in rows)
    indirect_calls = sum(int(row["indirect_call_sites"]) for row in rows)
    key_rows = {str(row["va"]): row for row in rows if row["va"] in KEY_VAS}

    lines = [
        "# TMNF physics function inventory",
        "",
        "## Scope",
        "",
        f"- Static functions reached from the four roots: {len(rows):,}",
        f"- Maximum minimum depth: {max_depth}",
        f"- Total instructions: {instruction_count:,}",
        f"- Unresolved indirect call sites: {indirect_calls:,}",
        (
            "- MSVC folded COMDAT helpers can have many map aliases at one VA. "
            "For those addresses, the inventory records Ghidra's primary demangled "
            "label and the matching final map row; the object module is not unique "
            "source provenance."
        ),
        "",
        "## Library breakdown",
        "",
    ]
    for library, count in library_counts.most_common():
        lines.append(f"- `{library}`: {count:,} functions")

    lines.extend(["", "## Object-module breakdown", ""])
    for module, count in module_counts.most_common():
        lines.append(f"- `{module}`: {count:,} functions")

    lines.extend(["", "## Key functions and direct callees", ""])
    for va in KEY_VAS:
        row = key_rows[va]
        lines.extend(
            [
                f"### `{va}` {row['demangled_name']}",
                "",
                f"- Mangled: `{row['mangled_name']}`",
                f"- Prototype: `{row['prototype']}`",
                f"- Module: `{row['module']}`",
                f"- Size: {int(row['size_bytes']):,} bytes",
                f"- Direct mapped callees: {int(row['direct_callees']):,}",
                "",
            ]
        )
        callees = sorted(
            edges_by_caller[va], key=lambda edge: int(edge["callee_va"], 16)
        )
        if callees:
            for edge in callees:
                lines.append(
                    f"- `{edge['callee_va']}` {edge['callee_name']}"
                )
        else:
            lines.append("- None")
        lines.append("")

    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_directory", type=Path)
    parser.add_argument("analysis_directory", type=Path)
    args = parser.parse_args()

    rows = number_rows(read_csv(args.raw_directory / "function_inventory.csv"))
    edges = read_csv(args.raw_directory / "call_edges.csv")
    controls = read_csv(args.raw_directory / "fp_control_sites.csv")

    shutil.copyfile(
        args.raw_directory / "function_inventory.csv",
        args.analysis_directory / "function_inventory.csv",
    )
    write_fp_report(args.analysis_directory / "fp_report.md", rows, controls)
    write_inventory_summary(
        args.analysis_directory / "inventory_summary.md", rows, edges
    )


if __name__ == "__main__":
    main()
