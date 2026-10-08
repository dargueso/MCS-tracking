"""mcstracking: tracker for mesoscale convective systems and convective storms
in gridded precipitation and brightness-temperature (or OLR) fields.

Optimised version of the A. Prein MCS-tracking algorithm, adapted to WRF output
or any netCDF with hourly rain and OLR on a 2-D lat/lon grid.
"""
__version__ = "2.0.0"

from .constants import const
from .config import load_config
from . import default_config
from .tracking import (
    MCStracking,
    olr_to_tb,
    label_objects,
    calc_grid_distance_area,
    calculate_area_objects,
    remove_small_short_objects,
)

__all__ = [
    "MCStracking", "olr_to_tb", "label_objects", "calc_grid_distance_area",
    "calculate_area_objects", "remove_small_short_objects", "const",
    "load_config", "default_config", "__version__",
]
