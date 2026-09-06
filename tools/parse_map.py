#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path


SYMBOL_RE = re.compile(
    r"^\s*0001:[0-9A-Fa-f]{8}\s+(\S+)\s+([0-9A-Fa-f]{8})\s+(.+?)\s*$"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("map_file", type=Path)
    parser.add_argument("output_tsv", type=Path)
    args = parser.parse_args()

    rows: list[tuple[str, str, int, str]] = []
    with args.map_file.open("r", encoding="latin-1") as map_file:
        for line in map_file:
            match = SYMBOL_RE.match(line)
            if match is None:
                continue

            mangled_name, va_text, suffix = match.groups()
            suffix_fields = suffix.split()
            module = suffix_fields[-1]
            flags = suffix_fields[:-1]
            rows.append(
                (
                    f"0x{int(va_text, 16):08X}",
                    mangled_name,
                    int("f" in flags),
                    module,
                )
            )

    with args.output_tsv.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, dialect="excel-tab", lineterminator="\n")
        writer.writerow(("va", "mangled_name", "is_function", "module"))
        writer.writerows(rows)


if __name__ == "__main__":
    main()
