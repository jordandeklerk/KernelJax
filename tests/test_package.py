"""Initialization tests for the KernelJax package."""

import kerneljax


def test_package_imports():
    assert kerneljax is not None


def test_version_is_exposed():
    assert isinstance(kerneljax.__version__, str)
    assert kerneljax.__version__
