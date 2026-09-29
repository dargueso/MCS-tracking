#!/usr/bin/env python
"""
Download EURADCLIM hourly accumulations from the KNMI Open Data API, plus the
OPERA radar database used for the distance-to-radar mask.

The platform serves one zip per month (~0.5-2.9 GB). Nothing here depends on
that: every listed
file whose name mentions a year in syear..eyear is fetched, and
make_radar_input.py indexes whatever arrives -- zips or loose .h5 -- by the
timestamp inside each HDF5 filename.

    python download_euradclim.py --list    # show what the API lists, fetch nothing
    python download_euradclim.py           # fetch, skipping files already present

Needs an API key: $KNMI_API_KEY, or ~/.knmi_api_key, or the shared anonymous
key (often exhausted: it is rate-limited for all its users together).

HTTP goes through curl, as in obs_download_utils: the python `requests` in the
MCStracking environment does not trust the certificate chain seen from this
machine, while the system curl does.
"""

import os
import re
import sys
import json
import time
import random
import logging
import argparse
import subprocess
from urllib.parse import urlencode

import radar_config as cfg
import obs_download_utils as util

NRETRIES = 5


def api_key():
    key = os.environ.get("KNMI_API_KEY")
    if not key and os.path.exists(os.path.expanduser("~/.knmi_api_key")):
        with open(os.path.expanduser("~/.knmi_api_key")) as fh:
            key = fh.read().strip()
    if not key:
        logging.warning("no personal KNMI key; using the shared anonymous one")
        key = cfg.knmi_anonymous_key
    return key


def curl(url, fileout=None, key=None, max_time=120):
    """Run curl; returns (http code, body text or None when writing to a file)."""
    cmd = ["curl", "-sS", "-L", "--max-time", str(max_time), "-w", "\n%{http_code}"]
    if key:
        cmd += ["-H", f"Authorization: {key}"]
    if fileout:
        cmd += ["-o", fileout]
    res = subprocess.run(cmd + [url], capture_output=True, text=True, check=False)
    out = res.stdout.rsplit("\n", 1)
    code = int(out[-1]) if out[-1].strip().isdigit() else 0
    return code, (None if fileout else out[0])


def call(url, key, params=None):
    """GET a JSON API endpoint, backing off on the rate limit."""
    if params:
        url = f"{url}?{urlencode(params)}"
    for attempt in range(1, 9):
        code, body = curl(url, key=key)
        if code == 200:
            return json.loads(body)
        limited = code in (403, 429) and "Rate Limit" in (body or "")
        if limited or code >= 500 or code == 0:
            wait = min(120, 10 * attempt) * (1 + random.random())
            logging.warning("%s (%d), waiting %.0f s", (body or "").strip()[:80], code, wait)
            time.sleep(wait)
            continue
        raise RuntimeError(f"{code} {(body or '')[:200]} for {url}\n"
                           "Check knmi_dataset / knmi_version in radar_config.py "
                           "against the dataset page on dataplatform.knmi.nl.")
    raise RuntimeError(f"gave up on {url} after repeated rate limiting")


def files_url():
    return f"{cfg.knmi_api}/datasets/{cfg.knmi_dataset}/versions/{cfg.knmi_version}/files"


def list_files(key):
    files, token = [], None
    while True:
        params = {"maxKeys": 500}
        if token:
            params["nextPageToken"] = token
        page = call(files_url(), key, params)
        files += page.get("files", [])
        token = page.get("nextPageToken")
        if not page.get("isTruncated") or not token:
            return files


def wanted(name):
    """Files of the requested years (a monthly archive, or an hourly file)."""
    years = [int(y) for y in re.findall(r"(?<!\d)(20\d\d)", name)]
    return any(cfg.syear <= y <= cfg.eyear for y in years)


def download(name, size, key):
    fileout = f"{cfg.path_rad_raw}/{name}"
    if os.path.exists(fileout) and (not size or os.path.getsize(fileout) == size):
        logging.info("%s already present, skipping", name)
        return True
    part = f"{fileout}.part"
    for attempt in range(1, NRETRIES + 1):
        # ask for a fresh temporary URL each time: it may expire mid-download
        url = call(f"{files_url()}/{name}/url", key)["temporaryDownloadUrl"]
        code, _ = curl(url, fileout=part, max_time=6 * 3600)
        try:
            if code != 200:
                raise IOError(f"http {code}")
            if size and os.path.getsize(part) != size:
                raise IOError(f"size {os.path.getsize(part)} != listed {size}")
            with open(part, "rb") as fh:
                head = fh.read(4)
            if head not in (b"PK\x03\x04", b"\x89HDF"):
                raise IOError(f"not a zip or HDF5 file (starts {head!r})")
        except (IOError, OSError) as err:
            logging.warning("attempt %d/%d %s: %s", attempt, NRETRIES, name, err)
            time.sleep(30 * attempt)
            continue
        os.replace(part, fileout)
        logging.info("got %s (%.2f GB)", name, os.path.getsize(fileout) / 1e9)
        return True
    if os.path.exists(part):
        os.remove(part)
    logging.error("GIVING UP on %s", name)
    return False


def opera_database():
    if os.path.exists(cfg.opera_db):
        return
    part = f"{cfg.opera_db}.part"
    code, _ = curl(cfg.opera_db_url, fileout=part)
    with open(part) as fh:
        json.load(fh)                            # fail here, not in the mask script
    if code != 200:
        raise RuntimeError(f"OPERA database: http {code}")
    os.replace(part, cfg.opera_db)
    logging.info("got OPERA radar database")


def main():
    par = argparse.ArgumentParser()
    par.add_argument("--list", action="store_true", help="list only")
    args = par.parse_args()
    util.start_logger("download_euradclim")
    os.makedirs(cfg.path_rad_raw, exist_ok=True)
    opera_database()

    key = api_key()
    files = list_files(key)
    logging.info("%d files listed for %s v%s", len(files), cfg.knmi_dataset, cfg.knmi_version)
    todo = [f for f in files if wanted(f["filename"])]
    if args.list:
        # show the wanted files, or a sample of all if the year filter matched
        # nothing (which means the naming is not what wanted() expects)
        for f in todo or files[:20]:
            print(f"{f['filename']:70s} {f.get('size', 0) / 1e9:8.2f} GB")
        logging.info("%d of them in %d-%d", len(todo), cfg.syear, cfg.eyear)
        return
    if len(todo) > 1000:
        logging.warning("%d files: the dataset is served as hourly files, not "
                        "archives. This works, but takes a while.", len(todo))
    failed = [f["filename"] for f in todo if not download(f["filename"], f.get("size"), key)]
    if failed:
        logging.error("%d failed: %s", len(failed), failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
