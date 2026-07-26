Vendored from https://github.com/ChristophReich1996/SMURF (unofficial PyTorch
inference port of SMURF, Stone et al. CVPR 2021), licensed CC-BY 4.0 — see
`LICENSE` in this directory. Files unmodified from upstream: `__init__.py`,
`_raft.py`, `smurf.py`.

Why vendored instead of pip-installed: the upstream repo has no
`setup.py`/`pyproject.toml`, so it isn't pip-installable; it's 3 small files
meant to be copied in.

Used by `video/flow/smurf_infer.py`, which wraps `raft_smurf()` with the
correct flow-direction and channel-order handling for this project's
`ytvis_ft15k.pt` checkpoint (see that module's docstring for why those
matter — this upstream repo demonstrates neither).
