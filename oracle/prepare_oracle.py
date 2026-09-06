#!/usr/bin/env python3
"""Configure the already-downloaded ModLoader and isolated Wine user data."""

from pathlib import Path
import os
import getpass
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PREFIX = Path(os.environ.get("TMNF_WINEPREFIX", ROOT / "oracle/wineprefix"))
WINE_USER = os.environ.get("TMNF_WINE_USER", getpass.getuser())
TMLOADER = ROOT / "third_party/TMLoader"
DOCUMENTS = PREFIX / "drive_c/users" / WINE_USER / "TMNFDocuments"


def main() -> None:
    settings = TMLOADER / "database/TmForever/products/TmForever/settings.yaml"
    settings.write_text('install: "C:/TmNationsForever"\n')

    profile = TMLOADER / "database/TmForever/profiles/default.yaml"
    profile.write_text("program:\n  id: TmForever\nmods:\n  - id: TMInterface\n")

    plugin_target = DOCUMENTS / "TMInterface/Plugins/Python_Link.as"
    plugin_target.parent.mkdir(parents=True, exist_ok=True)
    # Socket bridge provided by Linesight:
    # https://github.com/Linesight-RL/linesight
    shutil.copyfile(
        ROOT / "third_party/linesight/trackmania_rl/tmi_interaction/Python_Link.as",
        plugin_target,
    )
    (DOCUMENTS / "TMInterface/config.txt").write_text(
        "\n".join(
            (
                "set autologin 1",
                "set skip_map_load_screens true",
                "set unfocused_fps_limit false",
                "set disable_forced_camera true",
                "set autorewind false",
                "set auto_reload_plugins false",
                "",
            )
        )
    )

    track_target = DOCUMENTS / "TmForever/Tracks/Challenges/Official Maps/A01-Race.Challenge.Gbx"
    track_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        PREFIX
        / "drive_c/TmNationsForever/GameData/Tracks/Campaigns/Nations/White/A01-Race.Challenge.Gbx",
        track_target,
    )

    environment = os.environ | {"WINEPREFIX": str(PREFIX), "WINEDEBUG": "-all"}
    for registry_key in (
        r"HKCU\Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders",
        r"HKCU\Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders",
    ):
        subprocess.run(
            [
                "wine",
                "reg",
                "add",
                registry_key,
                "/v",
                "Personal",
                "/t",
                "REG_SZ",
                "/d",
                rf"C:\users\{WINE_USER}\TMNFDocuments",
                "/f",
            ],
            check=True,
            env=environment,
        )


if __name__ == "__main__":
    main()
