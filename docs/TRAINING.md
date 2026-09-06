# Training with local fixtures

Complete [local asset setup](LOCAL_ASSETS.md) first. The public source tree
ships no installed tracks. The track catalogue reads your generated
`oracle/tracks/manifest.txt`; each entry needs its matching track, vehicle and
route files.

The research Python environment uses Python 3.12:

```bash
uv venv --python python3.12 build/venv
uv pip install --python build/venv/bin/python -r requirements.txt
```

After generating A04 and building with your local physics image:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python build/venv/bin/python \
  -m tmnf_rl.train --track a04 --seed 1 --duration-minutes 10 \
  --thread-count 4 --run-id a04_first
```

Select one GPU explicitly and choose a CPU thread count appropriate to the
machine. Public builds do not reserve the original workstation's CPU IDs.
Set `TMNF_RESERVED_CPUS` to a comma-separated list and use `taskset` if you
want to protect cores for other work.

Training writes local runs below `build/runs/`. Evaluate with
`python -m tmnf_rl.evaluate <run-id-or-policy-path>` using the same environment
variables. The package also supports resume and reproducibility checks; use
individual commands' `--help` for their configuration options.

Vector observations are reused buffers: copy values that need to survive the
next step or reset. The environment uses same-step autoreset and provides
terminal observations and masks. Close native environments explicitly.
Game transfer requires a separate capture of the policy's generated inputs;
a simulator finish alone does not establish a matching game result.
