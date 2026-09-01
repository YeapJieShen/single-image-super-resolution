"""Torch-free support code: the LMDB cache, the vendored MATLAB resize, power state.

**Nothing is re-exported here, deliberately.** Both submodules are torch-free,
and :mod:`~sisr.utils.cache` must stay that way: ``parallel_build`` fans out
over a ``ProcessPoolExecutor``, and on spawn platforms each worker re-imports
the module tree holding its callable. A re-export would make importing *any*
member pull in *every* member, so one torch-dependent addition to this package
would silently put a multi-second torch import back into every build worker.

The guard lives in ``tests/utils/test_cache.py`` and covers this package, not just
the one module that happens to need it today.
"""
