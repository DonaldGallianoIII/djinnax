"""Shared: import paths for the gitignored reference clones + dep stubs."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # repo root
sys.path.insert(0, str(REPO_ROOT / "reference-engines" / "pgx"))
sys.path.insert(0, str(REPO_ROOT / "reference-engines" / "jumanji"))


class _Any:
    def __call__(self, *a, **k):
        return self

    def __getattr__(self, n):
        return self


_STUB_ROOTS = ("huggingface_hub", "tqdm", "esquilax")


class _StubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        m = types.ModuleType(spec.name)
        m.__path__ = []
        m.__getattr__ = lambda attr: _Any()
        return m

    def exec_module(self, module):
        pass


class _StubFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split(".")[0] in _STUB_ROOTS:
            return importlib.machinery.ModuleSpec(fullname, _StubLoader(), is_package=True)
        return None


if not any(isinstance(f, _StubFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _StubFinder())
