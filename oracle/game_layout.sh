# Source after setting ROOT and PREFIX. TMNF_WINE_USER overrides Wine's user.
WINE_USER="${TMNF_WINE_USER:-$(id -un)}"
if [[ -f "$PREFIX/drive_c/TmUnitedForever/TmForever.exe" ]]; then
    GAME_FLAVOR=united
    GAME="$PREFIX/drive_c/TmUnitedForever"
    USER_DOCUMENTS="$PREFIX/drive_c/users/$WINE_USER/Documents"
    USER_DIR="$USER_DOCUMENTS/TrackMania"
    TMLOADER="$ROOT/third_party/TMLoader_united/TMLoader.exe"
elif [[ -f "$PREFIX/drive_c/TmNationsForever/TmForever.exe" ]]; then
    GAME_FLAVOR=nations
    GAME="$PREFIX/drive_c/TmNationsForever"
    USER_DOCUMENTS="$PREFIX/drive_c/users/$WINE_USER/TMNFDocuments"
    USER_DIR="$USER_DOCUMENTS/TmForever"
    TMLOADER="$ROOT/third_party/TMLoader/TMLoader.exe"
else
    echo "missing Wine prefix with TmForever.exe: $PREFIX" >&2
    exit 2
fi
export GAME_FLAVOR GAME USER_DOCUMENTS USER_DIR TMLOADER
