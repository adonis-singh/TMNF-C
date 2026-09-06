#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GHIDRA_HEADLESS="/home/adityas/tools/ghidra_12.1.3_PUBLIC/support/analyzeHeadless"
EXE="$ROOT/oracle/wineprefix/drive_c/TmNationsForever/TmForever.exe"
MAP="$ROOT/oracle/wineprefix/drive_c/TmNationsForever/TmForever.map"
PROJECT_DIRECTORY="$ROOT/ghidra_proj"
PROJECT_NAME="TMNF"
SYMBOLS="$PROJECT_DIRECTORY/tmnf_symbols.tsv"
RAW_OUTPUT="$PROJECT_DIRECTORY/export"
SCRIPT_DIRECTORY="$ROOT/tools/ghidra_scripts"

mkdir -p "$PROJECT_DIRECTORY" "$RAW_OUTPUT" "$ROOT/analysis"
python "$ROOT/tools/parse_map.py" "$MAP" "$SYMBOLS"

"$GHIDRA_HEADLESS" "$PROJECT_DIRECTORY" "$PROJECT_NAME" \
    -import "$EXE" \
    -overwrite \
    -noanalysis \
    -scriptPath "$SCRIPT_DIRECTORY" \
    -postScript ApplyMapSymbols.java "$SYMBOLS"

"$GHIDRA_HEADLESS" "$PROJECT_DIRECTORY" "$PROJECT_NAME" \
    -process TmForever.exe \
    -analysisTimeoutPerFile 14400 \
    -scriptPath "$SCRIPT_DIRECTORY" \
    -preScript ConfigureAnalysis.java \
    -postScript ExportPhysicsInventory.java "$SYMBOLS" "$RAW_OUTPUT"

python "$ROOT/tools/generate_reports.py" "$RAW_OUTPUT" "$ROOT/analysis"
