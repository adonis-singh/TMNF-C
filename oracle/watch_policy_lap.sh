#!/usr/bin/env bash
# Launch TMNF visibly on the user's display and play the trained policy's lap
# through TMInterface at 1x speed. Unlike launch_game.sh this does NOT use
# Xvfb: the window appears on $DISPLAY. Rendering stays on llvmpipe (CPU) so
# no GPU is touched.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:-8484}"
INPUTS="${2:-$ROOT/oracle/results/policy_lap_inputs.bin}"
PREFIX="$ROOT/oracle/wineprefix"
TMLOADER="$ROOT/third_party/TMLoader/TMLoader.exe"
X11_INPUT="$ROOT/oracle/x11_key.py"

if [[ -z "${DISPLAY:-}" ]]; then
    echo "DISPLAY is not set; run from a graphical session" >&2
    exit 2
fi

export WINEPREFIX="$PREFIX"
export WINEDEBUG=-all
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe

cd "${TMLOADER%/*}"
wine "$TMLOADER" run TmForever default "/configstring=set custom_port $PORT" &
GAME_PID=$!

for _ in {1..60}; do
    if ss -ltn | rg -q "127\.0\.0\.1:$PORT "; then
        break
    fi
    sleep 1
done
ss -ltn | rg -q "127\.0\.0\.1:$PORT "

python3 "$X11_INPUT" --window "TrackMania Modded Forever" --key grave

"$ROOT/third_party/venv/bin/python" "$ROOT/oracle/capture_policy_lap.py" \
    --port "$PORT" \
    --inputs "$INPUTS" \
    --output /tmp/watch_policy_lap_discard.bin

echo
echo "Lap done. The game window stays open; close it when you are finished,"
echo "or press Ctrl+C here to kill it."
wait "$GAME_PID"
