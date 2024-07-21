from __future__ import annotations

import importlib.metadata

import mergeplg as m


def test_version():
    assert importlib.metadata.version("mergeplg") == m.__version__
