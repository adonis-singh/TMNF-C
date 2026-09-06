#!/usr/bin/env python3
"""Validate trace framing and summarize the first record in every file."""

from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[2]
TRACE_DIR = ROOT / "oracle/traces"


def read_u32(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<I", data, offset)[0], offset + 4


def summarize(path: Path) -> tuple[int, int, int, int]:
    data = path.read_bytes()
    if data[:8] != b"TMNFTRC1":
        raise RuntimeError(f"{path}: bad magic")
    function_va, offset = read_u32(data, 8)
    record_count, offset = read_u32(data, offset)
    first_input_size = 0
    first_output_size = 0
    for record_index in range(record_count):
        _, offset = read_u32(data, offset)
        n_in, n_out = struct.unpack_from("<HH", data, offset)
        offset += 4
        for buffer_index in range(n_in + n_out):
            _, offset = read_u32(data, offset)
            _, offset = read_u32(data, offset)
            length, offset = read_u32(data, offset)
            if record_index == 0:
                if buffer_index < n_in:
                    first_input_size += length
                else:
                    first_output_size += length
            offset += length
    if offset != len(data):
        raise RuntimeError(f"{path}: {len(data) - offset} trailing bytes")
    return function_va, record_count, first_input_size, first_output_size


def main() -> None:
    paths = sorted(TRACE_DIR.glob("*.bin"))
    if not paths:
        raise RuntimeError(f"no traces in {TRACE_DIR}")
    for path in paths:
        function_va, count, input_size, output_size = summarize(path)
        print(
            f"{function_va:08X} {path.stem[9:]}: records={count} "
            f"first_input={input_size} first_output={output_size}"
        )


if __name__ == "__main__":
    main()
