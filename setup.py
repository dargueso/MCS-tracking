# Shim for editable installs with setuptools < 64 (no PEP 660), e.g. the
# Python 3.8 environment in MCStracking.yml:  pip install -e . --no-build-isolation
# All metadata is in pyproject.toml.
from setuptools import setup

setup()
