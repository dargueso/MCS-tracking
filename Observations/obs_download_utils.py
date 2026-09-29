#!/usr/bin/env python
"""
Shared helpers for the IMERG / MERGIR downloads: Earthdata-authenticated
fetching, retries and month bookkeeping.

Authentication is done by shelling out to curl with -n (read ~/.netrc) and a
per-worker cookie jar. Earthdata redirects every data request to
urs.earthdata.nasa.gov and back, and a single shared cookie jar gets corrupted
when several workers write to it at once, so each worker gets its own.
"""

import os
import logging
import subprocess
import tempfile
import random
import shutil
import threading
import time

import pandas as pd

import obs_config as cfg


def daterange(year, month):
    """All days in a given year/month."""
    return pd.date_range(f"{year}-{month:02d}-01", periods=pd.Period(
        f"{year}-{month:02d}").days_in_month, freq="D")


def months_to_do():
    """(year, month) pairs requested in the configuration."""
    return [(y, m) for y in range(cfg.syear, cfg.eyear + 1) for m in cfg.months]


_MASTER_JAR = os.path.join(tempfile.gettempdir(), f"urs_cookies_master_{os.getpid()}")


def warm_up(url):
    """Establish the Earthdata session once, before any worker starts.

    The first request from an empty jar goes through the full URS redirect
    handshake. When several workers do that at the same instant the server
    refuses most of them with a 401, and they then sit in backoff for a minute
    or more before recovering. Doing the handshake once here and handing every
    worker a copy of the resulting jar avoids the whole episode.
    """
    probe = os.path.join(tempfile.gettempdir(), f"urs_warmup_{os.getpid()}")
    ok = fetch(url, probe, jar=_MASTER_JAR)
    if os.path.exists(probe):
        os.remove(probe)
    if ok:
        logging.info("Earthdata session established")
    else:
        logging.warning("could not pre-authenticate; workers will each handshake")
    return ok


def cookie_jar():
    """A private cookie jar for the calling worker.

    Keyed on process AND thread: the downloads run on a thread pool, so a
    pid-only name would still have every worker writing the same jar. Earthdata
    bounces each request through urs.earthdata.nasa.gov and back, and
    concurrent writes to one jar corrupt the session cookie, which comes back
    as sporadic 401s.
    """
    jar = os.path.join(tempfile.gettempdir(),
                       f"urs_cookies_{os.getpid()}_{threading.get_ident()}")
    if not os.path.exists(jar) and os.path.exists(_MASTER_JAR):
        shutil.copy(_MASTER_JAR, jar)  # start from the already-authenticated session
    return jar


# netCDF-4/HDF5 and netCDF classic file signatures.
_MAGIC = (b"\x89HDF", b"CDF\x01", b"CDF\x02")


def looks_like_data(path):
    """True if the file starts with a netCDF/HDF5 signature.

    GES DISC answers some requests with HTTP 200 and a body that is not data at
    all - an OPeNDAP error document, or a truncated response when the server is
    under load. Without this check those are accepted as successful downloads
    and only fail later, at the point of reading, where they were being dropped
    silently: ~12% of one MERGIR month went missing that way.
    """
    try:
        with open(path, "rb") as fh:
            return fh.read(4) in _MAGIC
    except OSError:
        return False


def fetch(url, fileout, jar=None):
    """Download one URL to fileout, with retries. Returns True on success.

    The file is written to a .part file and only moved into place once curl
    reports success, so an interrupted run never leaves a truncated file that a
    later run would mistake for a complete one.
    """
    jar = jar or cookie_jar()
    part = f"{fileout}.part"
    for attempt in range(1, cfg.nretries + 1):
        try:
            res = subprocess.run(
                ["curl", "-sS", "-g", "-L", "-n",
                 "-b", jar, "-c", jar,
                 "--max-time", str(cfg.timeout),
                 "-w", "%{http_code}",
                 "-o", part, url],
                capture_output=True, text=True, check=False)
            code = (res.stdout or "").strip()[-3:]
            if res.returncode == 0 and code == "200" and os.path.getsize(part) > 0:
                if looks_like_data(part):
                    os.replace(part, fileout)
                    return True
                # 200 but not a data file. In practice this is the GES DISC
                # HTML login page: the URS session cookie has expired, so the
                # data request is redirected to the login form and that page is
                # served with a 200. Waiting does not help - the session has to
                # be re-established - so drop the stale cookie jars and let the
                # next attempt do a fresh handshake from ~/.netrc.
                with open(part, "rb") as fh:
                    head = fh.read(200).decode("utf-8", "replace")
                # Only this worker's jar is dropped: tested, workers with
                # independent sessions fail at the same rate as workers sharing
                # one, so there is nothing to gain by invalidating the others.
                stale = "<html" in head.lower() or "<!doctype html" in head.lower()
                if stale and os.path.exists(jar):
                    try:
                        os.remove(jar)
                    except OSError:
                        pass
                logging.warning("attempt %d/%d %s %s", attempt, cfg.nretries,
                                "session expired, re-authenticating" if stale
                                else "non-data payload: " + " ".join(head.split())[:80],
                                os.path.basename(url))
            else:
                logging.warning("attempt %d/%d http=%s rc=%d %s",
                                attempt, cfg.nretries, code, res.returncode,
                                os.path.basename(url))
        except Exception as err:  # network hiccup, DNS, ...
            logging.warning("attempt %d/%d failed (%s) %s",
                            attempt, cfg.nretries, err, os.path.basename(url))
        # GES DISC throttles bursts: too many concurrent requests come back as
        # 503, and sometimes as a spurious 401 once the redirect is refused.
        # Both clear after a pause, so back off generously and jitter the wait
        # so the workers do not all retry in lockstep.
        time.sleep(min(90, 5 * attempt ** 2) * (1 + random.random()))
    if os.path.exists(part):
        os.remove(part)
    logging.error("GIVING UP on %s", url)
    return False


def subset_bounds():
    """Region plus margin, as (lat0, lat1, lon0, lon1)."""
    return (cfg.lat_min - cfg.margin, cfg.lat_max + cfg.margin,
            cfg.lon_min - cfg.margin, cfg.lon_max + cfg.margin)


def already_done(fileout):
    """True if this month can be skipped."""
    return os.path.exists(fileout) and not cfg.overwrite


def start_logger(name):
    logger = logging.getLogger()
    if not logger.handlers:
        logging.basicConfig(
            format="%(asctime)s | %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
        fh = logging.FileHandler(f"{name}.log", mode="a")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
        logger.addHandler(fh)
    return logger
