"""Resolution of the tracking configuration module.

The tracker reads its thresholds and options from a plain Python module. Which
module is decided once, by ``load_config()``:

1. the environment variable ``MCS_CONFIG``, if set, names either a module
   importable from ``sys.path`` (``MCS_CONFIG=mcs_config_sens``) or a path to
   a ``.py`` file (``MCS_CONFIG=/path/to/my_config.py``). For a file the
   directory is put on ``sys.path`` first, so the module stays importable by
   name (``from my_base_config import *`` inside it works, and joblib workers
   re-import it by name);
2. otherwise ``mcs_config``, from ``sys.path`` or from the working directory;
3. otherwise the packaged reference configuration, ``mcstracking.default_config``.

The module object is cached by the import system, so every caller sees the same
object and attributes set at run time (``cfg.rain_source`` by the WRF driver,
for instance) reach the tracker.
"""
import importlib
import logging
import os
import sys

_log = logging.getLogger("mcstracking")


def load_config(spec=None):
    """Return the configuration module (see the module docstring)."""
    spec = spec or os.environ.get("MCS_CONFIG")
    if spec and (spec.endswith(".py") or os.sep in spec):
        path = os.path.abspath(spec)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"MCS_CONFIG={spec!r}: no such file")
        directory = os.path.dirname(path)
        name = os.path.splitext(os.path.basename(path))[0]
        if directory not in sys.path:
            sys.path.insert(0, directory)
        mod = importlib.import_module(name)
        if os.path.abspath(getattr(mod, "__file__", "")) != path:
            raise ImportError(
                f"MCS_CONFIG={spec!r} resolved to {mod.__file__}: another module "
                f"named {name!r} shadows it on sys.path")
    elif spec:
        mod = importlib.import_module(spec)
    else:
        try:
            mod = importlib.import_module("mcs_config")
        except ModuleNotFoundError:
            if os.path.isfile("mcs_config.py"):
                return load_config(os.path.abspath("mcs_config.py"))
            from . import default_config as mod
            _log.info("no mcs_config found: using mcstracking.default_config")
    _log.info("tracking configuration: %s", getattr(mod, "__file__", mod.__name__))
    return mod
