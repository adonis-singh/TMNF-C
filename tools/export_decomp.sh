#!/usr/bin/env bash
# Re-attaches to the already-analyzed Ghidra project and exports decompilation,
# disassembly, and vtables. No re-analysis.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GHIDRA_HEADLESS="/home/adityas/tools/ghidra_12.1.3_PUBLIC/support/analyzeHeadless"
MAP="$ROOT/oracle/wineprefix/drive_c/TmNationsForever/TmForever.map"
PROJECT_DIRECTORY="$ROOT/ghidra_proj"
PROJECT_NAME="TMNF"
INVENTORY="$ROOT/analysis/function_inventory.csv"
OUT="$ROOT/ghidra_proj/export"
SCRIPT_DIRECTORY="$ROOT/tools/ghidra_scripts"

"$GHIDRA_HEADLESS" "$PROJECT_DIRECTORY" "$PROJECT_NAME" \
    -process TmForever.exe \
    -noanalysis \
    -scriptPath "$SCRIPT_DIRECTORY" \
    -postScript ExportDecompilation.java "$INVENTORY" "$OUT" \
    -postScript ExportVtables.java "$MAP" "$OUT"
