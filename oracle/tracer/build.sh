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
i686-w64-mingw32-gcc -std=c11 -O2 -Wall -Wextra -Werror \
    -shared -static-libgcc -Wl,--kill-at \
    "${SOURCES[@]}" -o "$OUT/TMNFTracer.asi"

file "$OUT/TMNFTracer.asi"
