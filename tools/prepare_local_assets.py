#!/usr/bin/env python3
"""Extract the required physics mask from your installed Forever packs."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packs', type=Path, required=True)
    p.add_argument('--dotnet', default=shutil.which('dotnet'))
    args = p.parse_args()
    if not (args.packs / 'packlist.dat').is_file():
        p.error('--packs must contain your installed packlist.dat and .pak files')
    if not args.dotnet:
        p.error('install .NET 10 SDK and provide --dotnet or put dotnet on PATH')
    local = ROOT / 'local'
    local.mkdir(exist_ok=True)
    candidates = local / 'mask-candidates.txt'
    candidates.write_text('TestMaterialHeight.tga\n')
    project = ROOT / 'tools/extract_game_assets/GbxTool/GbxTool.csproj'
    subprocess.run([args.dotnet, 'build', str(project), '-c', 'Release',
                    '--configfile', str(project.parents[1] / 'NuGet.config')], check=True)
    dll = project.parent / 'bin/Release/net10.0/GbxTool.dll'
    extracted = local / 'mask-extract'
    subprocess.run([args.dotnet, str(dll), 'extract', str(args.packs.resolve()),
                    str(extracted), 'TestMaterialHeight.tga', '--candidates', str(candidates)], check=True)
    files = sorted(extracted.rglob('TestMaterialHeight.tga'))
    if not files:
        p.error('installed packs did not yield TestMaterialHeight.tga')
    subprocess.run([sys.executable, str(ROOT / 'tools/prepare_game_mask.py'),
                    'extract', str(files[0]), str(local / 'game-mask.bin')], check=True)
    print('Local physics mask ready. Configure CMake with -DTMNF_GAME_MASK=' + str(local / 'game-mask.bin'))
    print('Track, vehicle and route fixtures still require the capture workflow in docs/LOCAL_ASSETS.md.')


if __name__ == '__main__':
    main()
