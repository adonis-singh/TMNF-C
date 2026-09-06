#!/usr/bin/env bash
set -u
cd /home/adityas/Projects/TMNF-C
export FIXER_TRACER_SRC=/home/adityas/Projects/TMNF-C-fixer/oracle/tracer
for t in oracle/tracks/*.tmnftrack; do
  n=$(basename "$t" .tmnftrack)
  if [ -f "/home/adityas/fixer-tools/tracks/$n.tmnftrack" ]; then echo "skip $n"; continue; fi
  echo "== $n $(date +%T)"
  timeout 600 python3 /home/adityas/fixer-tools/capture_fixer.py "$n" --mode track --probe --inputs /dev/null --output /dev/null 2>&1 | tail -1
  tail -1 "/home/adityas/fixer-tools/tracks/$n.tmnftrack.log"
done
echo ALL-DONE
