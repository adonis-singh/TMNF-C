#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-trace}"
if [[ $# -gt 1 || ("$MODE" != trace && "$MODE" != track && "$MODE" != route && "$MODE" != detect) ]]; then
    echo "usage: $0 [trace|track|route|detect]" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PREFIX="${TMNF_WINEPREFIX:-$ROOT/oracle/wineprefix}"
source "$ROOT/oracle/game_layout.sh"
DIST="$ROOT/third_party/dist"
ARCHIVE="$DIST/Ultimate-ASI-Loader-v9.7.4-NoPDB.zip"
ARCHIVE_SHA256="14b3a1ad018899571ac9aa01482977f3c6d49e6cba99f552d01c5acacd1315e1"

bash "$ROOT/oracle/tracer/build.sh" "$MODE"
mkdir -p "$DIST"
curl -fL \
    "https://github.com/ThirteenAG/Ultimate-ASI-Loader/releases/download/v9.7.4/Ultimate-ASI-Loader-NoPDB.zip" \
    -o "$ARCHIVE"
printf '%s  %s\n' "$ARCHIVE_SHA256" "$ARCHIVE" | sha256sum -c -
unzip -p "$ARCHIVE" dinput8.dll > "$DIST/ultimate-asi-binkw32.dll"

if [[ ! -f "$GAME/binkw32Hooked.dll" ]]; then
    mv "$GAME/binkw32.dll" "$GAME/binkw32Hooked.dll"
fi
cp "$DIST/ultimate-asi-binkw32.dll" "$GAME/binkw32.dll"
cp "$ROOT/oracle/tracer/build/TMNFTracer.asi" "$GAME/TMNFTracer.asi"
