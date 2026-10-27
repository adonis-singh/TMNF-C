#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-trace}"
if [[ $# -gt 1 || ("$MODE" != trace && "$MODE" != track && "$MODE" != route && "$MODE" != detect) ]]; then
    echo "usage: $0 [trace|track|route|detect]" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/oracle/tracer/build"

mkdir -p "$OUT"
if [[ "$MODE" == trace || "$MODE" == detect ]]; then
    TARGETS="$ROOT/oracle/tracer/targets.c"
    if [[ "$MODE" == detect ]]; then
        TARGETS="$ROOT/oracle/tracer/detect_targets.c"
    fi
    SOURCES=(
        "$ROOT/oracle/tracer/tracer.c"
        "$TARGETS"
        "$ROOT/oracle/tracer/broadphase_capture.c"
        "$ROOT/oracle/tracer/detect_capture.c"
        "$ROOT/oracle/tracer/response_capture.c"
        "$ROOT/oracle/tracer/model6_capture.c"
    )
elif [[ "$MODE" == track ]]; then
    SOURCES=("$ROOT/oracle/tracer/track_dump.c")
else
    SOURCES=("$ROOT/oracle/tracer/route_dump.c")
fi
CC="${CC:-i686-w64-mingw32-gcc}"
FLAGS=(-std=c11 -O2 -Wall -Wextra -Werror -shared -static-libgcc -Wl,--kill-at)
CC_VERSION="$("$CC" -dumpfullversion -dumpversion 2>/dev/null || "$CC" -dumpversion)"
mapfile -t TRACER_SOURCES < <(find "$ROOT/oracle/tracer" -maxdepth 1 -type f \
    \( -name '*.c' -o -name '*.h' \) -print | sort)
FINGERPRINT="$({ printf '%s\n' "$MODE" "$CC_VERSION" "${FLAGS[@]}"; \
    sha256sum "${TRACER_SOURCES[@]}"; } | sha256sum | cut -d' ' -f1)"
CACHE="$OUT/TMNFTracer.$MODE.$FINGERPRINT.asi"
LOCK="$OUT/.build-$MODE.lock"
exec 9>"$LOCK"
flock 9
if [[ -s "$CACHE" ]]; then
    cp -f "$CACHE" "$OUT/TMNFTracer.asi"
    file "$OUT/TMNFTracer.asi"
    exit 0
fi
TEMP="$OUT/.TMNFTracer.$MODE.$$.$FINGERPRINT.tmp.asi"
trap 'rm -f "$TEMP"' EXIT
"$CC" "${FLAGS[@]}" "${SOURCES[@]}" -o "$TEMP"
mv -f "$TEMP" "$OUT/TMNFTracer.asi"
cp -f "$OUT/TMNFTracer.asi" "$CACHE"

file "$OUT/TMNFTracer.asi"
