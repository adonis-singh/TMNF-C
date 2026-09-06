#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 PORT [TRACK_NAME]" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="$1"
TRACK_NAME="${2:-A01-Race}"
PREFIX="${TMNF_WINEPREFIX:-$ROOT/oracle/wineprefix}"
DISPLAY_NUMBER="${TMNF_XVFB_DISPLAY:-98}"
source "$ROOT/oracle/game_layout.sh"
X11_INPUT="$ROOT/oracle/x11_key.py"
TRACE_DIR="${TMNF_TRACE_HOST_DIR:-$ROOT/oracle/traces}"
TRACK_DIR="${TMNF_TRACK_DIR:-$ROOT/oracle/tracks}"
ROUTE_DIR="${TMNF_ROUTE_DIR:-$ROOT/oracle/routes}"
CHALLENGE="$USER_DIR/Tracks/Challenges/Official Maps/$TRACK_NAME.Challenge.Gbx"

if [[ ! -f "$CHALLENGE" ]]; then
    echo "missing installed challenge: $CHALLENGE" >&2
    exit 2
fi

export WINEPREFIX="$PREFIX"
export WINEDEBUG=-all
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
mkdir -p "$TRACE_DIR"
mkdir -p "$TRACK_DIR"
mkdir -p "$ROUTE_DIR"
export TMNF_TRACE_DIR
TMNF_TRACE_DIR="Z:${TRACE_DIR//\//\\}"
export TMNF_TRACK_PATH
TMNF_TRACK_PATH="Z:${TRACK_DIR//\//\\}\\$TRACK_NAME.tmnftrack"
export TMNF_ROUTE_PATH
TMNF_ROUTE_PATH="Z:${ROUTE_DIR//\//\\}\\$TRACK_NAME.tmnfroute"
export TMNF_TRACK_SHA256
TMNF_TRACK_SHA256="$(sha256sum "$CHALLENGE" | cut -d' ' -f1)"

exec xvfb-run -n "$DISPLAY_NUMBER" -s "-screen 0 1024x768x24" bash -c '
    set -euo pipefail
    cd "${1%/*}"
    wine "$1" run TmForever default "/configstring=set custom_port $2" &

    for _ in {1..60}; do
        if ss -ltn | rg -q "127\.0\.0\.1:$2 "; then
            break
        fi
        sleep 1
    done
    ss -ltn | rg -q "127\.0\.0\.1:$2 "

    python3 "$3" --window "TrackMania Modded Forever" --key grave
    wineserver -w
' bash "$TMLOADER" "$PORT" "$X11_INPUT"
