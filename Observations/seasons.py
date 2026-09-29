#!/usr/bin/env python
"""
Month selection shared by the evaluation plots (radar and gauges), so every
script takes the same options and names its output the same way.

    --season ASON        named season (default ASON, the study season)
    --all-months         the whole year, same as --season ANN
    --months 6 7 8       any other set of months

The three are mutually exclusive. The tag (ASON, ANN, ...; m6-7-8 for a custom
set) goes into every output filename, so runs for different months sit side
by side instead of overwriting each other.
"""

SEASONS = {
    "ASON": (8, 9, 10, 11),
    "ANN": tuple(range(1, 13)),
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}
DEFAULT = "ASON"


def add_args(parser):
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--season", choices=list(SEASONS), default=None,
                     help=f"named season (default {DEFAULT})")
    grp.add_argument("--all-months", action="store_true",
                     help="evaluate the whole year (same as --season ANN)")
    grp.add_argument("--months", nargs="+", type=int, default=None,
                     help="custom list of months, e.g. --months 6 7 8")


def resolve(args):
    """(months, tag) from parsed arguments."""
    if args.all_months:
        return list(SEASONS["ANN"]), "ANN"
    if args.months:
        months = sorted(set(args.months))
        if any(m < 1 or m > 12 for m in months):
            raise SystemExit(f"--months: invalid month in {args.months}")
        for name, sel in SEASONS.items():
            if sorted(sel) == months:
                return list(sel), name
        return months, "m" + "-".join(map(str, months))
    name = args.season or DEFAULT
    return list(SEASONS[name]), name
