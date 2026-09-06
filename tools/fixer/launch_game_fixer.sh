#!/usr/bin/env bash
# Copy of oracle/launch_game.sh bound to oracle/wineprefix_fixer, Xvfb :91.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 PORT [TRACK_NAME]" >&2
    exit 2
fi

ROOT="/home/adityas/Projects/TMNF-C"
PORT="$1"
TRACK_NAME="${2:-A01-Race}"
PREFIX="$ROOT/oracle/wineprefix_fixer"
TMLOADER="$ROOT/third_party/TMLoader/TMLoader.exe"
X11_INPUT="$ROOT/oracle/x11_key.py"
TRACE_DIR="${TMNF_TRACE_HOST_DIR:-/home/adityas/fixer-tools/traces}"
TRACK_DIR="/home/adityas/fixer-tools/tracks"
ROUTE_DIR="/home/adityas/fixer-tools/routes"
CHALLENGE="$PREFIX/drive_c/users/adityas/TMNFDocuments/TmForever/Tracks/Challenges/Official Maps/$TRACK_NAME.Challenge.Gbx"

if [[ ! -f "$CHALLENGE" ]]; then
    echo "missing installed challenge: $CHALLENGE" >&2
    exit 2
fi

export WINEPREFIX="$PREFIX"
export WINEDEBUG=-all
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export CUDA_VISIBLE_DEVICES=
mkdir -p "$TRACE_DIR" "$TRACK_DIR" "$ROUTE_DIR"
export TMNF_TRACE_DIR
TMNF_TRACE_DIR="Z:${TRACE_DIR//\//\\}"
export TMNF_TRACK_PATH
TMNF_TRACK_PATH="Z:${TRACK_DIR//\//\\}\\$TRACK_NAME.tmnftrack"
export TMNF_ROUTE_PATH
TMNF_ROUTE_PATH="Z:${ROUTE_DIR//\//\\}\\$TRACK_NAME.tmnfroute"
export TMNF_TRACK_SHA256
TMNF_TRACK_SHA256="$(sha256sum "$CHALLENGE" | cut -d' ' -f1)"

exec xvfb-run -n 89 -s "-screen 0 1024x768x24" bash -c '
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
