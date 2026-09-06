#!/usr/bin/env bash
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
INSTALL_DIR="$ROOT/third_party/dotnet"
INSTALL_SCRIPT="$ROOT/third_party/dotnet-install.sh"

mkdir -p "$ROOT/third_party"
curl -fsSL https://dot.net/v1/dotnet-install.sh -o "$INSTALL_SCRIPT"
bash "$INSTALL_SCRIPT" --version 10.0.400 --install-dir "$INSTALL_DIR"
