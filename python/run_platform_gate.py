#!/usr/bin/env python3
"""Run the `python_platform` gate against a frozen copy of the session.

The gate used to import `python/tmnf_rl` and load `build/libtmnf_physics.so`
from the live checkout, so another agent saving a file or rebuilding the
library mid-session changed the hashes the reproduce/resume tests compare, and
tracks added to `oracle/tracks/manifest.txt` during the run changed what the
multi-track tests iterated (review R7, footgun F31). This runner:

1. keeps a detached git worktree at `build/Testing/platform_tree` checked out
   at HEAD (committed fixtures only, so a manifest track whose fixtures are not
   in git is visibly skipped, F32), copies the live `python/` sources and the
   live `build/libtmnf_physics.so` / `build/export_viewer_scene` into it, and
   runs pytest from there;
2. writes the full output to `build/Testing/python_platform_<stamp>.log`, one
   file per run, never overwritten;
3. hashes the live library and package before and after and prints a NOTE when
   they changed during the session (the gate result stands: it tested the copy).

Invoked by ctest (`tests/CMakeLists.txt`) and usable by hand:
`python python/run_platform_gate.py [-- pytest args]`.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCES = ("tmnf_rl", "test_tmnf_rl.py", "test_tmnf_env.py")
BUILD_ARTIFACTS = ("libtmnf_physics.so", "export_viewer_scene")


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_sha256(package: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def live_hashes(build: Path) -> dict[str, str]:
    hashes = {
        "physics_sha256": sha256_file(build / "libtmnf_physics.so"),
        "code_sha256": package_sha256(ROOT / "python" / "tmnf_rl"),
    }
    if (build / "libtmnf_cuda.so").is_file():
        hashes["cuda_sha256"] = sha256_file(build / "libtmnf_cuda.so")
    return hashes


def sync_tree(tree: Path, build: Path) -> str:
    """Detached worktree at HEAD plus the live python sources and binaries."""
    head = git("rev-parse", "HEAD")
    if not (tree / ".git").exists():
        tree.parent.mkdir(parents=True, exist_ok=True)
        if tree.exists():
            shutil.rmtree(tree)
        git("worktree", "prune")
        git("worktree", "add", "--detach", str(tree), head)
    else:
        git("checkout", "--quiet", "--detach", "--force", head, cwd=tree)
    # Drop every untracked file under python/ (stale copies, __pycache__).
    git("clean", "-fdq", "--", "python", cwd=tree)
    for name in PYTHON_SOURCES:
        source = ROOT / "python" / name
        target = tree / "python" / name
        if source.is_dir():
            shutil.rmtree(target, ignore_errors=True)
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(source, target)
    (tree / "build").mkdir(exist_ok=True)
    for name in BUILD_ARTIFACTS:
        shutil.copy2(build / name, tree / "build" / name)
    # CUDA binding/parity tests must exercise the library that was built. An
    # old copy must not make a CPU-only build appear to have CUDA coverage.
    cuda_source = build / "libtmnf_cuda.so"
    cuda_target = tree / "build" / "libtmnf_cuda.so"
    if cuda_source.is_file():
        shutil.copy2(cuda_source, cuda_target)
    else:
        cuda_target.unlink(missing_ok=True)
    return head


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_platform_gate")
    parser.add_argument("--build", type=Path, default=ROOT / "build")
    parser.add_argument("pytest_args", nargs="*", help="extra pytest arguments (after --)")
    args = parser.parse_args(argv)
    build = args.build.resolve()
    testing = build / "Testing"
    testing.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    log_path = testing / f"python_platform_{stamp}.log"
    tree = testing / "platform_tree"

    before = live_hashes(build)
    head = sync_tree(tree, build)
    tree_hashes = {
        **live_hashes(tree / "build"),
        "code_sha256": package_sha256(tree / "python" / "tmnf_rl"),
    }
    assert tree_hashes == before, (tree_hashes, before)
    command = [
        sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider",
        str(tree / "python" / "test_tmnf_rl.py"), *args.pytest_args,
    ]
    env = {**os.environ, "PYTHONPATH": str(tree / "python"), "CUDA_VISIBLE_DEVICES": "0"}
    header = [
        f"python_platform gate {stamp}",
        f"log: {log_path}",
        f"tree: {tree} (HEAD {head} + live python/ and build binaries)",
        f"physics_sha256: {before['physics_sha256']}",
        f"cuda_sha256: {before.get('cuda_sha256', 'not built')}",
        f"code_sha256: {before['code_sha256']}",
        f"command: {' '.join(command)}",
        "",
    ]
    with log_path.open("w", encoding="utf-8") as log:
        def emit(line: str) -> None:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
            log.write(line + "\n")
            log.flush()

        for line in header:
            emit(line)
        process = subprocess.Popen(
            command, cwd=tree, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        assert process.stdout is not None
        for line in process.stdout:
            emit(line.rstrip("\n"))
        code = process.wait()
        after = live_hashes(build)
        if after != before:
            changed = ", ".join(
                f"{key} {before.get(key, 'missing')[:8]} -> {after.get(key, 'missing')[:8]}"
                for key in sorted(before.keys() | after.keys())
                if before.get(key) != after.get(key)
            )
            emit(f"NOTE: live tree changed during the session ({changed}); the gate tested the copy above")
        emit(f"pytest exit {code}; log retained at {log_path}")
    return code


if __name__ == "__main__":
    sys.exit(main())
