"""Build the pinned GBX.NET patch used by GbxDump, once per checkout.

The source archive is hash checked. A local NuGet package keeps the small
compatibility patch reviewable without vendoring the entire upstream project.
It preserves the game's flags for a malformed final legacy block and consumes
only its exact three-null-reference padding signature. Other parse failures
remain errors. Original maps and replay files are never rewritten.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
COMMIT = '975272fa085ba32dc65080c6904a1ebeee903201'
SOURCE_SHA256 = '3bbf9b30612b29dec2ed80c768aab24132c8c789dc27ea42d0a4ba1e81c0a288'
PACKAGE_VERSION = '2.4.4.2'
PATCH = HERE / 'gbx_legacy_clip.patch'
BUILD = REPO / 'third_party/gbx-net-compat'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_package(dotnet):
    """Return a verified local package; never modify the installed NuGet cache."""
    packages = BUILD / 'packages'
    packages.mkdir(parents=True, exist_ok=True)
    package = packages / f'GBX.NET.{PACKAGE_VERSION}.nupkg'
    # Keep earlier package receipts intact for frozen validation runtimes.
    receipt = BUILD / f'package-{PACKAGE_VERSION}.json'
    patch_hash = sha256(PATCH)
    with (BUILD / '.build.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if package.exists() and receipt.exists():
            expected = json.loads(receipt.read_text())
            if (expected['commit'] == COMMIT and expected['patch_sha256'] == patch_hash
                    and expected['package_sha256'] == sha256(package)):
                return package
            raise RuntimeError('GBX.NET compatibility patch changed; bump PACKAGE_VERSION before rebuilding')

        archive = BUILD / f'upstream-{COMMIT}.tar.gz'
        if not archive.exists():
            temporary = archive.with_suffix('.partial')
            urllib.request.urlretrieve(
                f'https://codeload.github.com/BigBang1112/gbx-net/tar.gz/{COMMIT}', temporary)
            if sha256(temporary) != SOURCE_SHA256:
                raise ValueError('GBX.NET source archive hash differs')
            temporary.replace(archive)
        if sha256(archive) != SOURCE_SHA256:
            raise ValueError('GBX.NET source archive hash differs')

        source = BUILD / f'source-{patch_hash[:12]}'
        if not source.exists():
            with tempfile.TemporaryDirectory(prefix='extract-', dir=BUILD) as folder:
                staging = Path(folder)
                with tarfile.open(archive) as stream:
                    stream.extractall(staging, filter='data')
                extracted = staging / f'gbx-net-{COMMIT}'
                subprocess.run(['patch', '--batch', '--fuzz=0', '-p1', '-i', str(PATCH)],
                               cwd=extracted, check=True, stdout=sys.stderr)
                project = extracted / 'Src/GBX.NET/GBX.NET.csproj'
                text = project.read_text(encoding='utf-8-sig')
                frameworks = 'net10.0;net9.0;net8.0;netstandard2.0'
                if text.count(frameworks) != 1:
                    raise ValueError('pinned GBX.NET project framework list changed')
                project.write_text(text.replace(frameworks, 'net10.0'))
                shutil.copyfile(HERE / 'NuGet.config', extracted / 'NuGet.config')
                # This dependency build needs only the upstream feeds, not its
                # own package feed whose relative path belongs to GbxDump.
                config = (extracted / 'NuGet.config').read_text()
                config = '\n'.join(line for line in config.splitlines()
                                   if 'tmnf-local' not in line)
                (extracted / 'NuGet.config').write_text(config + '\n')
                extracted.rename(source)

        subprocess.run([str(dotnet), 'pack', str(source / 'Src/GBX.NET/GBX.NET.csproj'),
                        '-c', 'Release', '-o', str(packages),
                        f'-p:PackageVersion={PACKAGE_VERSION}', '-p:GeneratePackageOnBuild=false',
                        '-p:IsTrimmable=false', '-p:IsAotCompatible=false', '-v:q'],
                       check=True, stdout=sys.stderr)
        metadata = dict(commit=COMMIT, source_sha256=SOURCE_SHA256,
                        patch_sha256=patch_hash, package_sha256=sha256(package),
                        package_version=PACKAGE_VERSION)
        temporary = receipt.with_suffix('.partial')
        temporary.write_text(json.dumps(metadata, indent=2) + '\n')
        temporary.replace(receipt)
    return package


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dotnet', required=True, type=Path)
    print(ensure_package(parser.parse_args().dotnet))
