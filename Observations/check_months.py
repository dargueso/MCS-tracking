#!/usr/bin/env python
"""
Report how complete each downloaded month is, and optionally delete the ones
that are short so a re-run picks them up again.

Months are skipped by filename, so an incomplete month is never revisited
unless its file is removed. Run this after every download session.

    python check_months.py            # report only
    python check_months.py --delete   # also remove the incomplete months
"""

import os
import sys
import glob

import pandas as pd
import xarray as xr

import obs_config as cfg


def expected(tag, per_day):
    year, month = (int(x) for x in tag.split("-"))
    return pd.Period(f"{year}-{month:02d}").days_in_month * per_day


def check(pattern, per_day, label, delete):
    bad = []
    for fin in sorted(glob.glob(pattern)):
        tag = os.path.basename(fin).rsplit("_", 1)[-1].replace(".nc", "")
        try:
            with xr.open_dataset(fin) as dset:
                steps = dset.sizes["time"]
                attr = dset.attrs.get("granules") or dset.attrs.get("files", "?/?")
        except Exception as err:
            print(f"  {tag}  UNREADABLE ({err})")
            bad.append(fin)
            continue
        want = expected(tag, per_day)
        ok = steps == want
        print(f"  {tag}  {attr:>12s}  steps {steps}/{want}  "
              f"{'ok' if ok else 'INCOMPLETE, lost ' + str(want - steps)}")
        if not ok:
            bad.append(fin)

    print(f"{label}: {len(bad)} incomplete")
    if bad and delete:
        for fin in bad:
            os.remove(fin)
        print(f"  deleted {len(bad)} file(s); re-run the downloader to redo them")
    elif bad:
        print("  re-run with --delete to remove them, then re-run the downloader")
    return bad


def main():
    delete = "--delete" in sys.argv
    print("IMERG (48 half-hourly granules per day)")
    a = check(f"{cfg.path_imerg}/IMERG_30MIN_PR_*.nc", 48, "IMERG", delete)
    print("\nMERGIR (24 hourly files per day, 2 steps each -> 48 steps)")
    b = check(f"{cfg.path_mergir}/MERGIR_30MIN_TB_*.nc", 48, "MERGIR", delete)
    return 1 if (a or b) and not delete else 0


if __name__ == "__main__":
    sys.exit(main())
