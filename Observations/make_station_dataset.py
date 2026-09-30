#!/usr/bin/env python
"""
Build the combined, quality-controlled AEMET rain-gauge dataset: 10-minute,
2011-2020, with the clock-hour dataset derived from it and the Arnau daily
record alongside.

Sources and roles (see station_config.py):
  HyMEX 10-min      raw real-time AWS data, not quality controlled
  Balearic 10-min   AEMET historical archive, cuenca B
  Arnau daily       AEMET validated daily totals and maxima, with flags

For every station and day each 10-min source gets a tier:

  A  verified   Arnau has a validated record derived from the 10-min data
                (ID_FLAG_P = 1, quality flags 0/1), the day is complete, and
                the 10-min data reproduce Arnau's daily total, maximum 10-min
                and maximum 60-min amounts within rounding (0.2 mm + 5%).
                Arnau is computed from the same record, so a larger
                difference means AEMET corrected the day and the raw copy is
                wrong. Automatic checks must also pass.
  B  automatic  no usable Arnau record for the day (2020; stations not in
                Arnau; Arnau not derived from the 10-min record, or flagged
                unchecked, or accumulated), and the automatic checks pass:
                10-min <= 50 mm, day <= 400 mm, no stuck sensor, no isolated
                spike; a partial day must not exceed Arnau's total or maxima.
                A partial day that DOES reproduce Arnau is tier A: amounts
                cannot be negative, so the gaps held no rain.
  rejected      Arnau flags the day doubtful, the data disagree with Arnau,
                or an automatic check fails. The whole day is removed.

Where both 10-min sources have a day, the Balearic archive is used if it
passes, else HyMEX. Only accepted days carry data in the output.

Time: both sources stamp the END of each 10-min interval; the output is
stamped at the START (see station_config.source_stamp for how this was
established). Hourly values are clock hours, 13:00 = 13:00-14:00 UTC, valid
only when all six 10-min values are.

Outputs (station_config): file_10min, file_01h, file_arnau.

    python make_station_dataset.py
"""

import os
import io
import glob
import logging
import zipfile
import subprocess

import numpy as np
import pandas as pd
import xarray as xr

import station_config as cfg
import obs_download_utils as util

STEPS = 144                     # 10-min steps per day
T0 = pd.Timestamp(f"{cfg.syear}-01-01")
DAYS = pd.date_range(T0, f"{cfg.eyear}-12-31", freq="D")
NDAY = DAYS.size


###########################################################
# Reading
###########################################################

def dms(value, lon):
    """AEMET DDMMSS[o] coordinates -> degrees (o: 1 east, 2 west, longitude only)."""
    s = str(int(value)).zfill(7 if lon else 6)
    sign = 1.0
    if lon:
        sign = -1.0 if s[-1] == "2" else 1.0
        s = s[:-1]
    d, m, sec = int(s[:-4]), int(s[-4:-2]), int(s[-2:])
    return sign * (d + m / 60 + sec / 3600)


def unpack_arnau():
    os.makedirs(cfg.path_arnau, exist_ok=True)
    for name in cfg.arnau_archives:
        if not os.path.exists(f"{cfg.path_arnau}/{name}.txt"):
            subprocess.run(["unrar", "x", "-o+", "-inul", f"{cfg.path_arnau_rar}/{name}.rar",
                            f"{cfg.path_arnau}/"], check=True)
            logging.info("unpacked %s", name)


def load_arnau():
    cols = ["INDICATIVO", "ANO", "MES", "DIA", "NOMBRE", "ALTITUD", "LONGITUD", "LATITUD",
            "P24", "PMAX10", "PMAX20", "PMAX30", "PMAX60", "PMAX2H", "PMAX6H", "PMAX12H",
            "ID_FLAG_Q1", "ID_FLAG_Q2", "ID_FLAG_E", "ID_FLAG_P"]
    arn = pd.concat([pd.read_csv(f"{cfg.path_arnau}/{n}.txt", sep=";", encoding="latin1",
                                 usecols=cols, dtype={"INDICATIVO": str})
                     for n in cfg.arnau_archives])
    arn = arn[(arn.ANO >= cfg.syear) & (arn.ANO <= cfg.eyear)]
    arn["date"] = pd.to_datetime(dict(year=arn.ANO, month=arn.MES, day=arn.DIA))
    arn = arn.drop_duplicates(["INDICATIVO", "date"], keep="last")
    logging.info("Arnau: %d station-days, %d stations, %s to %s", len(arn),
                 arn.INDICATIVO.nunique(), arn.date.min().date(), arn.date.max().date())
    return arn


def synoptic_map(arn):
    """{synoptic (WMO) code: climatological code (INDICATIVO)}.

    HyMEX names a station by its synoptic code where it has one and by its
    climatological code otherwise, and some stations appear under BOTH in the
    same year. Arnau and the Balearic archive use the climatological code. The
    pairs come from the Arnau station lists (readme.rar) and the Balearic
    master file. Where a synoptic code has had several climatological codes
    (station moved or renamed), the one with most Arnau days is taken.
    """
    rar = f"{cfg.path_arnau_rar}/readme.rar"
    subprocess.run(["unrar", "x", "-o+", "-inul", rar, f"{cfg.path_arnau}/"], check=True)
    pairs = [pd.read_csv(f, sep=";", encoding="latin1", dtype=str)[["INDICATIVO", "IND_SYN"]]
             for f in glob.glob(f"{cfg.path_arnau}/Maestro*.csv")]
    pairs.append(pd.read_csv(cfg.balear_master, sep=";", encoding="latin1",
                             dtype=str)[["INDICATIVO", "IND_SYN"]])
    p = pd.concat(pairs).dropna()
    p["syn"] = p.IND_SYN.str.strip().str.zfill(5)
    p = p[p.syn != "00000"].drop_duplicates(["syn", "INDICATIVO"])
    ndays = arn.INDICATIVO.value_counts()
    p["n"] = p.INDICATIVO.map(ndays).fillna(0)
    p = p.sort_values("n", ascending=False).drop_duplicates("syn")
    return dict(zip(p.syn, p.INDICATIVO.str.strip()))


def load_hymex(syn2ind):
    """{station: 10-min array on the output axis}, START-stamped, canonical codes."""
    nt = NDAY * STEPS
    out = {}
    conflicts = {}
    for fin in sorted(glob.glob(f"{cfg.path_hymex}/prec_10min_all_stations_*.pkl")):
        d = pd.read_pickle(fin)
        d = d.loc[:, [c for c in d.columns if isinstance(c, str) and c.strip()]]
        d = d.loc[:, ~d.columns.duplicated()]
        # Rows off the 10-min grid (:15/:45 and a few odd minutes in 2011-2015,
        # from a 15-min station, AR01) fall in the same slot as a real :10/:40
        # value and, empty for every other station, would overwrite it with NaN.
        on_grid = (d.index.minute % 10 == 0) & (d.index.second == 0)
        if not on_grid.all():
            logging.info("HyMEX %s: dropped %d rows off the 10-min grid", os.path.basename(fin),
                         (~on_grid).sum())
            d = d.loc[on_grid]
        # END stamp -> START stamp, then position on the output axis
        pos = ((d.index - pd.Timedelta(minutes=10)) - T0) // pd.Timedelta(minutes=10)
        pos = np.asarray(pos)
        ok = (pos >= 0) & (pos < nt)
        # climatological codes first, so a synoptic-coded copy only fills gaps
        for code in sorted(d.columns, key=lambda c: c in syn2ind):
            vals = d[code].values[ok].astype("float32")
            if not np.isfinite(vals).any():
                continue
            canon = syn2ind.get(code, code)
            arr = out.setdefault(canon, np.full(nt, np.nan, "float32"))
            # only finite values are written, so a missing value never erases one
            good = np.isfinite(vals)
            p, vals = pos[ok][good], vals[good]
            old = arr[p]
            both = np.isfinite(old)
            if both.any():
                n = conflicts.setdefault(canon, [0, 0])
                n[0] += int(both.sum())
                n[1] += int((np.abs(old[both] - vals[both]) > 0.05).sum())
            arr[p] = np.where(both, old, vals)
        logging.info("HyMEX %s: %d columns", os.path.basename(fin), d.shape[1])
    if conflicts:
        tot = sum(v[0] for v in conflicts.values())
        bad = sum(v[1] for v in conflicts.values())
        logging.info("HyMEX: %d stations present under both codes; %d overlapping values, "
                     "%d (%.3f%%) differ", len(conflicts), tot, bad, 100 * bad / max(tot, 1))
    return out


def load_balear():
    nt = NDAY * STEPS
    parts = []
    for fin in sorted(glob.glob(f"{cfg.path_balear}/historico_diezmin_cuenca_B_*.csv")):
        b = pd.read_csv(fin, sep=";", encoding="latin1", decimal=",")
        b.columns = [c.strip() for c in b.columns]
        b = b.rename(columns={b.columns[1]: "ANO"})           # AÑO, whatever the encoding
        parts.append(b[["INDICATIVO", "ANO", "MES", "DIA", "HORA", "MINUTO", "PREC"]])
    b = pd.concat(parts)
    t = pd.to_datetime(dict(year=b.ANO, month=b.MES, day=b.DIA, hour=b.HORA, minute=b.MINUTO))
    pos = np.asarray((t - pd.Timedelta(minutes=10) - T0) // pd.Timedelta(minutes=10))
    ok = (pos >= 0) & (pos < nt) & b.PREC.notna().values
    b, pos = b[ok], pos[ok]
    out = {}
    for code, idx in b.groupby("INDICATIVO").indices.items():
        arr = np.full(nt, np.nan, "float32")
        arr[pos[idx]] = b.PREC.values[idx]
        out[str(code).strip()] = arr
    logging.info("Balearic archive: %d stations", len(out))
    return out


def metadata(codes, arn):
    """lat, lon, alt, name per station, from the best available source."""
    meta = {}
    # lowest priority first; later sources overwrite
    for fin in glob.glob(f"{cfg.path_hrly_meta}/*.txt"):
        head = {}
        with open(fin, errors="replace") as fh:
            for _ in range(5):
                line = fh.readline()
                if ":" in line:
                    k, v = line.split(":", 1)
                    head[k.strip()] = v.strip()
        try:
            meta[head["Idema"]] = (float(head["lat"]), float(head["lon"]),
                                   float(head["alt"]), head.get("ubi", ""))
        except (KeyError, ValueError):
            pass
    first = arn.drop_duplicates("INDICATIVO")
    for r in first.itertuples():
        meta[r.INDICATIVO] = (dms(r.LATITUD, False), dms(r.LONGITUD, True),
                              float(r.ALTITUD), str(r.NOMBRE).strip())
    with zipfile.ZipFile(cfg.hymex_zip) as zf:
        st = pd.read_csv(io.BytesIO(zf.read("0_Documentation/AWS_AEMET-stations.csv")),
                         encoding="latin1")
    for r in st.itertuples():
        meta[r.Code] = (r.gLat, r.gLon, float(r.alt), str(r.Station_name))
    bm = pd.read_csv(cfg.balear_master, sep=";", encoding="latin1", decimal=",")
    for r in bm.itertuples():
        meta[r.INDICATIVO] = (float(r.LATITUD), float(r.LONGITUD), float(r.ALTITUD),
                              str(r.NOMBRE).strip())
    missing = [c for c in codes if c not in meta]
    if missing:
        logging.warning("%d stations without coordinates, dropped: %s", len(missing),
                        ", ".join(sorted(missing)))
    return {c: meta[c] for c in codes if c in meta}


###########################################################
# Quality control
###########################################################

def day_stats(x):
    """Per-day total, max 10-min, max 60-min (within the day), valid steps. x: (nday, 144)."""
    valid = np.isfinite(x).sum(1)
    v = np.where(np.isfinite(x), x, 0.0)
    total = v.sum(1)
    max10 = v.max(1)
    c = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(v, 1)], 1)
    max60 = (c[:, 6:] - c[:, :-6]).max(1)
    return total, max10, max60, valid


def auto_ok(x):
    """Automatic checks per day. x: (nday, 144), NaN missing."""
    v = np.where(np.isfinite(x), x, 0.0)
    ok = (v.max(1) <= cfg.max_10min) & (v.sum(1) <= cfg.max_day)
    # stuck sensor: stuck_steps identical values >= stuck_min_value in a row
    same = (v[:, 1:] == v[:, :-1]) & (v[:, 1:] >= cfg.stuck_min_value)
    k = cfg.stuck_steps - 1
    c = np.concatenate([np.zeros((v.shape[0], 1)), np.cumsum(same, 1)], 1)
    ok &= ~((c[:, k:] - c[:, :-k]) >= k).any(1)
    # isolated spike: big value, nothing else within +-spike_window steps
    w = cfg.spike_window
    pad = np.pad(v, ((0, 0), (w, w)))
    cs = np.concatenate([np.zeros((v.shape[0], 1)), np.cumsum(pad, 1)], 1)
    around = cs[:, 2 * w + 1:] - cs[:, :-(2 * w + 1)] - v
    ok &= ~((v >= cfg.spike_value) & (around == 0)).any(1)
    return ok


def arnau_table(arn, code):
    """Arnau fields for one station on the day axis (NaN where absent)."""
    a = arn[arn.INDICATIVO == code].set_index("date").reindex(DAYS)
    out = {}
    for k in ("P24", "PMAX10", "PMAX60"):
        raw = a[k].values.astype(float)
        out[k + "_raw"] = raw
        out[k] = np.where(raw == -3, 0.0, np.where(raw < 0, np.nan, raw / 10.0))
    for k in ("ID_FLAG_Q1", "ID_FLAG_Q2", "ID_FLAG_P", "ID_FLAG_E"):
        out[k] = a[k].values
    return out


REASONS = {}     # why days end in tier B, counted over all stations and sources


def count(name, mask):
    REASONS[name] = REASONS.get(name, 0) + int(mask.sum())


def tiers(x10, arn_st, hourly=False):
    """Tier per day for one source of one station. x10: (nday, 144).

    hourly=True is the rule set for the clock-hour product: the maximum 10-min
    amount is not part of the verification, since a 10-min timing or rounding
    difference inside an hour does not make the hourly totals wrong. The daily
    total and the maximum 60-min amount must still match.
    """
    total, max10, max60, valid = day_stats(x10)
    has = valid > 0
    tier = np.where(has, cfg.TIER_B, cfg.TIER_NONE).astype("i1")
    auto = auto_ok(x10)
    if arn_st is None:
        count("station not in Arnau", has & auto)
        return np.where(has & ~auto, cfg.TIER_REJ_AUTO, tier).astype("i1")

    q1, q2 = arn_st["ID_FLAG_Q1"], arn_st["ID_FLAG_Q2"]
    present = np.isfinite(q1)
    bad = present & (np.isin(q1, cfg.arnau_bad_flags) | np.isin(q2, cfg.arnau_bad_flags))
    have = np.isfinite(arn_st["P24"]) & np.isfinite(arn_st["PMAX60"])
    if not hourly:
        have &= np.isfinite(arn_st["PMAX10"])
    checkable = (present & np.isin(q1, cfg.arnau_good_flags)
                 & np.isin(q2, cfg.arnau_good_flags) & have)
    # Arnau derived from this same 10-min record: a mismatch means AEMET
    # corrected the day, so the raw copy is wrong. Arnau from another
    # instrument (P = 0/2): a match still verifies, a mismatch proves nothing.
    derived = arn_st["ID_FLAG_P"] == 1
    tol = lambda ref: cfg.tol_abs + cfg.tol_rel * np.nan_to_num(ref)
    complete = valid >= cfg.min_steps_day
    match = ((np.abs(total - arn_st["P24"]) <= tol(arn_st["P24"]))
             & (np.abs(max60 - arn_st["PMAX60"]) <= tol(arn_st["PMAX60"])))
    # a partial day cannot hold more rain, or a bigger maximum, than AEMET's
    # validated record for the whole day
    exceeds = ((total > arn_st["P24"] + tol(arn_st["P24"]))
               | (max60 > arn_st["PMAX60"] + tol(arn_st["PMAX60"])))
    if not hourly:
        match &= np.abs(max10 - arn_st["PMAX10"]) <= tol(arn_st["PMAX10"])
        exceeds |= max10 > arn_st["PMAX10"] + tol(arn_st["PMAX10"])

    # A day that reproduces Arnau is verified even with gaps: amounts cannot be
    # negative, so matching the validated total means the gaps held no rain.
    tier = np.where(has & checkable & match, cfg.TIER_A, tier)
    tier = np.where(has & checkable & derived & complete & ~match, cfg.TIER_REJ_ARNAU, tier)
    tier = np.where(has & checkable & ~complete & exceeds, cfg.TIER_REJ_ARNAU, tier)
    tier = np.where(has & bad, cfg.TIER_REJ_ARNAU, tier)
    tier = np.where(has & ~auto & (tier != cfg.TIER_REJ_ARNAU), cfg.TIER_REJ_AUTO, tier)
    b = tier == cfg.TIER_B
    after = DAYS > pd.Timestamp("2019-12-31")
    count("after Arnau ends (2020)", b & after)
    b &= ~after
    count("no Arnau record that day", b & ~present)
    b &= present
    count("Arnau unchecked (flag 10)", b & (np.isin(q1, (10,)) | np.isin(q2, (10,))))
    b &= ~(np.isin(q1, (10,)) | np.isin(q2, (10,)))
    count("Arnau value missing/accumulated", b & ~have)
    b &= have
    count("Arnau from another instrument and not matching (P=0/2)", b & ~derived)
    b &= derived
    count("incomplete day, below Arnau total (rain may be in the gaps)", b)
    return tier.astype("i1")


def build(codes, arn, hym, bal, hourly):
    """Combine the two 10-min sources station by station under one rule set.

    Returns prec (station, 10-min steps), day tier, day source, per-source
    tiers and the Balearic/HyMEX agreement. With hourly=True the hourly rules
    are used and, in verified days, missing 10-min steps are set to 0: the
    day reproduces AEMET's validated total, so the gaps held no rain, and
    every hour of the day becomes known.
    """
    REASONS.clear()
    arn_codes = set(arn.INDICATIVO)
    ns = len(codes)
    prec = np.full((ns, NDAY * STEPS), np.nan, "float32")
    tier = np.zeros((ns, NDAY), "i1")
    source = np.zeros((ns, NDAY), "i1")          # 1 HyMEX, 2 Balearic archive
    tier_by_src = {"hymex": np.zeros((ns, NDAY), "i1"), "balear": np.zeros((ns, NDAY), "i1")}
    agree = []
    for i, code in enumerate(codes):
        arn_st = arnau_table(arn, code) if code in arn_codes else None
        cands = []
        for key, src, data in (("balear", 2, bal), ("hymex", 1, hym)):   # priority order
            if code in data:
                x = data[code].reshape(NDAY, STEPS)
                t = tiers(x, arn_st, hourly=hourly)
                tier_by_src[key][i] = t
                cands.append((src, x, t))
        view = prec[i].reshape(NDAY, STEPS)
        for src, x, t in cands:
            take = np.isin(t, (cfg.TIER_A, cfg.TIER_B)) & (source[i] == 0)
            view[take] = x[take]
            tier[i][take] = t[take]
            source[i][take] = src
        # days no source could supply: keep the most informative rejection
        for src, x, t in cands:
            fill = (tier[i] == cfg.TIER_NONE) & (t != cfg.TIER_NONE)
            tier[i][fill] = t[fill]
        if hourly:
            ver = tier[i] == cfg.TIER_A
            view[ver] = np.where(np.isfinite(view[ver]), view[ver], 0.0)
        if len(cands) == 2:
            both = (tier_by_src["balear"][i] == cfg.TIER_A) & (tier_by_src["hymex"][i] == cfg.TIER_A)
            if both.any():
                xb = bal[code].reshape(NDAY, STEPS)[both]
                xh = hym[code].reshape(NDAY, STEPS)[both]
                both_ok = np.isfinite(xb) & np.isfinite(xh)
                if both_ok.any():
                    agree.append(np.mean(np.abs(xb[both_ok] - xh[both_ok]) < 0.05))
    return prec, tier, source, tier_by_src, agree, dict(REASONS)


def main():
    util.start_logger("make_station_dataset")
    os.makedirs(cfg.path_db, exist_ok=True)
    unpack_arnau()
    arn = load_arnau()
    syn2ind = synoptic_map(arn)
    hym = load_hymex(syn2ind)
    bal = load_balear()
    codes = sorted(set(hym) | set(bal))
    meta = metadata(sorted(set(codes) | set(arn.INDICATIVO)), arn)
    codes = [c for c in codes if c in meta]
    logging.info("%d stations with 10-min data (%d HyMEX, %d Balearic, %d both)", len(codes),
                 len(set(hym) & set(codes)), len(set(bal) & set(codes)),
                 len(set(hym) & set(bal) & set(codes)))
    p10, t10, s10, tb10, ag10, r10 = build(codes, arn, hym, bal, hourly=False)
    p1h, t1h, s1h, tb1h, ag1h, r1h = build(codes, arn, hym, bal, hourly=True)
    lines = summary("10-MIN PRODUCT", t10, tb10, ag10, r10)
    lines += summary("CLOCK-HOUR PRODUCT (hourly rules, gaps of verified days = 0)",
                     t1h, tb1h, ag1h, r1h)
    lines += hourly_gain(p10, t10, p1h, t1h)
    for line in lines:
        logging.info(line)
    with open(f"{cfg.path_db}/build_summary.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    lat = np.array([meta[c][0] for c in codes])
    lon = np.array([meta[c][1] for c in codes])
    alt = np.array([meta[c][2] for c in codes])
    names = np.array([meta[c][3] for c in codes], dtype=object)
    write_10min(codes, names, lat, lon, alt, p10, t10, s10)
    write_01h(codes, names, lat, lon, alt, p1h, t1h, s1h)
    daily = write_daily(codes, names, lat, lon, alt, p10, t10, p1h, t1h, s1h)
    write_arnau(arn, meta)
    write_stations(codes, names, lat, lon, alt, hym, bal, arn, t10, t1h, p1h, daily)
    write_readme(lines, codes)


def summary(title, tier, tier_by_src, agree, reasons):
    has = tier != cfg.TIER_NONE
    lines = [title, "station-days with data, by tier:"]
    for t, what in cfg.tier_meaning.items():
        if t != cfg.TIER_NONE:
            lines.append(f"  {t}  {(tier == t).sum() / has.sum():6.1%}  {what}")
    for key, tb in tier_by_src.items():
        h = tb != cfg.TIER_NONE
        if h.any():
            lines.append(f"  source {key:6s}: " + ", ".join(
                f"tier {t} {(tb == t).sum() / h.sum():.1%}" for t in (1, 2, 8, 9)))
    tot_b = sum(reasons.values())
    lines.append("why days are tier B (all sources):")
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {v / max(tot_b, 1):6.1%}  {k}")
    if agree:
        lines.append(f"  Balearic archive vs HyMEX on days both verify: "
                     f"{np.mean(agree):.2%} of 10-min values identical")
    years = DAYS.year.values
    ok = np.isin(tier, (cfg.TIER_A, cfg.TIER_B))
    lines.append("accepted station-days per year (% of stations x days):")
    lines.append("  " + "  ".join(f"{y}:{ok[:, years == y].mean():.0%}"
                                  for y in range(cfg.syear, cfg.eyear + 1)))
    return lines + [""]


def valid_hours(prec):
    x = prec.reshape(prec.shape[0], -1, 6)
    return np.isfinite(x).all(2)


def hourly_gain(p10, t10, p1h, t1h):
    """How much the hourly rules add over taking complete hours of the 10-min product."""
    a, b = valid_hours(p10), valid_hours(p1h)
    na, nb = a.sum(), b.sum()
    years = pd.date_range(T0, periods=a.shape[1], freq="h").year.values
    enough = lambda v: int(((v.reshape(v.shape[0], -1).mean(1)) >= 0.5).sum())
    lines = ["HOURLY GAIN from the hourly rules",
             f"  valid station-hours: {na:,} -> {nb:,} (+{(nb - na) / na:.1%})",
             f"  stations with >= 50% valid hours over {cfg.syear}-{cfg.eyear}: "
             f"{enough(a)} -> {enough(b)}",
             "  per year: " + "  ".join(f"{y}:{a[:, years == y].mean():.0%}->{b[:, years == y].mean():.0%}"
                                        for y in range(cfg.syear, cfg.eyear + 1))]
    return lines
def station_coords(codes, names, lat, lon, alt):
    return {"code": ("station", np.array(codes, dtype=object)),
            "name": ("station", names),
            "lat": ("station", lat.astype("float32"), {"units": "degrees_north"}),
            "lon": ("station", lon.astype("float32"), {"units": "degrees_east"}),
            "alt": ("station", alt.astype("float32"), {"units": "m"})}


def common_attrs():
    return {
        "institution_data": "AEMET (Agencia Estatal de Meteorologia)",
        "sources": "HyMEX AWS 10-min (real time, restricted to HyMeX projects); "
                   "AEMET historical 10-min archive, cuenca B; "
                   "AEMET climatological daily database with sub-daily maxima (Arnau)",
        "quality_control": "; ".join(f"{k}={v}" for k, v in cfg.tier_meaning.items()),
        "qc_settings": (f"tol {cfg.tol_abs} mm + {cfg.tol_rel:.0%}; max_10min {cfg.max_10min} mm; "
                        f"max_day {cfg.max_day} mm; stuck {cfg.stuck_steps} x >= "
                        f"{cfg.stuck_min_value} mm; spike >= {cfg.spike_value} mm isolated "
                        f"+-{cfg.spike_window} steps"),
        "source_codes": "1 HyMEX, 2 Balearic archive, 0 none",
        "history": "make_station_dataset.py (MCS-tracking/Observations)",
    }


def write_10min(codes, names, lat, lon, alt, prec, tier, source):
    times = pd.date_range(T0, periods=NDAY * STEPS, freq="10min")
    ds = xr.Dataset(
        {"prec": (("station", "time"), prec,
                  {"units": "mm", "long_name": "precipitation in the 10 min starting at time",
                   "cell_methods": "time: sum"}),
         "day_tier": (("station", "day"), tier, {"long_name": "quality tier of the day"}),
         "day_source": (("station", "day"), source, {"long_name": "source of the day"})},
        coords={"time": times, "day": DAYS, **station_coords(codes, names, lat, lon, alt)},
        attrs={**common_attrs(), "title": "AEMET 10-min precipitation, combined and QC'd",
               "time_convention": "START of the 10-min interval, UTC (sources stamp the end)"})
    enc = {"prec": {"zlib": True, "complevel": 4, "chunksizes": (1, NDAY * STEPS // 10)},
           "day_tier": {"zlib": True}, "day_source": {"zlib": True}}
    ds.to_netcdf(f"{cfg.file_10min}.tmp", encoding=enc)
    os.replace(f"{cfg.file_10min}.tmp", cfg.file_10min)
    logging.info("wrote %s", cfg.file_10min)


def write_01h(codes, names, lat, lon, alt, prec, tier, source):
    ns = len(codes)
    x = prec.reshape(ns, NDAY * 24, 6)
    hourly = np.where(np.isfinite(x).all(2), np.nansum(x, 2), np.nan).astype("float32")
    times = pd.date_range(T0, periods=NDAY * 24, freq="h")
    ds = xr.Dataset(
        {"prec": (("station", "time"), hourly,
                  {"units": "mm", "long_name": "precipitation in the clock hour starting at time",
                   "cell_methods": "time: sum",
                   "comment": "hourly rule set: days verified on the daily total and max "
                              "60-min amount (not the max 10-min); in verified days missing "
                              "10-min steps are 0 (the validated total shows they held no "
                              "rain); otherwise valid only if all six 10-min values are"}),
         "day_tier": (("station", "day"), tier), "day_source": (("station", "day"), source)},
        coords={"time": times, "day": DAYS, **station_coords(codes, names, lat, lon, alt)},
        attrs={**common_attrs(), "title": "AEMET clock-hour precipitation from the combined "
                                          "QC'd 10-min dataset",
               "time_convention": "START of the clock hour, UTC"})
    enc = {"prec": {"zlib": True, "complevel": 4, "chunksizes": (1, NDAY * 24)}}
    ds.to_netcdf(f"{cfg.file_01h}.tmp", encoding=enc)
    os.replace(f"{cfg.file_01h}.tmp", cfg.file_01h)
    logging.info("wrote %s", cfg.file_01h)


def write_daily(codes, names, lat, lon, alt, p10, t10, p1h, t1h, s1h):
    """Daily totals and maxima per station, in the manner of Arnau, from our own data.

    Day = 00-24 UTC, as in Arnau. Two families, each from its own product:
      from the 10-min product  max_10min, max_60min_sliding (60-min windows at
                               10-min steps within the day: the analogue of
                               Arnau's PMAX60) and the time of the 10-min maximum
      from the 1-h product     prec (daily total), max_1h (clock hour: the
                               quantity comparable with hourly model output)
                               and the hour of that maximum
    A value is written only if its day is complete in that product; otherwise
    NaN, with n_valid_10min / n_valid_1h kept. In verified days (tier 1)
    missing 10-min steps are 0, since the day reproduces AEMET's validated
    total, so verified days are always complete.
    """
    ns = len(codes)
    x = p10.reshape(ns, NDAY, STEPS).copy()
    ver = t10 == cfg.TIER_A
    x[ver] = np.where(np.isfinite(x[ver]), x[ver], 0.0)
    n10 = np.isfinite(x).sum(2)
    ok10 = (n10 == STEPS) & np.isin(t10, (cfg.TIER_A, cfg.TIER_B))
    v = np.where(np.isfinite(x), x, 0.0)
    max10 = np.where(ok10, v.max(2), np.nan)
    arg10 = np.where(ok10, v.argmax(2), -1)
    c = np.concatenate([np.zeros((ns, NDAY, 1)), np.cumsum(v, 2)], 2)
    max60 = np.where(ok10, (c[..., 6:] - c[..., :-6]).max(2), np.nan)

    h = p1h.reshape(ns, NDAY * 24, 6)
    hourly = np.where(np.isfinite(h).all(2), np.nansum(h, 2), np.nan).reshape(ns, NDAY, 24)
    n1h = np.isfinite(hourly).sum(2)
    ok1h = (n1h == 24) & np.isin(t1h, (cfg.TIER_A, cfg.TIER_B))
    hv = np.where(np.isfinite(hourly), hourly, 0.0)
    prec = np.where(ok1h, hv.sum(2), np.nan)
    max1h = np.where(ok1h, hv.max(2), np.nan)
    arg1h = np.where(ok1h, hv.argmax(2), -1)
    # a dry day has no time of maximum
    arg10 = np.where(max10 > 0, arg10, -1)
    arg1h = np.where(max1h > 0, arg1h, -1)

    f32 = lambda a: a.astype("float32")
    ds = xr.Dataset(
        {"prec": (("station", "day"), f32(prec),
                  {"units": "mm", "long_name": "daily total, 00-24 UTC", "from": "1-h product"}),
         "max_1h": (("station", "day"), f32(max1h),
                    {"units": "mm", "long_name": "maximum clock-hour amount of the day",
                     "from": "1-h product",
                     "comment": "clock hours, comparable with hourly model output"}),
         "hour_max_1h": (("station", "day"), arg1h.astype("i1"),
                         {"long_name": "start hour (UTC) of the maximum clock hour",
                          "comment": "-1 if dry or incomplete"}),
         "max_10min": (("station", "day"), f32(max10),
                       {"units": "mm", "long_name": "maximum 10-min amount of the day",
                        "from": "10-min product", "comment": "analogue of Arnau PMAX10"}),
         "time_max_10min": (("station", "day"), arg10.astype("i2"),
                            {"long_name": "start of the maximum 10-min interval, minutes/10 "
                                          "after 00 UTC (0-143)",
                             "comment": "-1 if dry or incomplete"}),
         "max_60min_sliding": (("station", "day"), f32(max60),
                               {"units": "mm", "long_name": "maximum 60-min amount of the day, "
                                "windows at 10-min steps within the day",
                                "from": "10-min product", "comment": "analogue of Arnau PMAX60"}),
         "n_valid_10min": (("station", "day"), n10.astype("i2"),
                           {"long_name": "valid 10-min steps (verified-day gaps counted as 0 mm)"}),
         "n_valid_1h": (("station", "day"), n1h.astype("i1"), {"long_name": "valid clock hours"}),
         "tier_10min": (("station", "day"), t10, {"long_name": "day tier, 10-min rules"}),
         "tier_1h": (("station", "day"), t1h, {"long_name": "day tier, hourly rules"}),
         "source_1h": (("station", "day"), s1h)},
        coords={"day": DAYS, **station_coords(codes, names, lat, lon, alt)},
        attrs={**common_attrs(), "title": "AEMET daily totals and maxima from the combined "
                                          "QC'd dataset (Arnau-style)",
               "day_convention": "00-24 UTC",
               "validity": "a value is present only if its day is complete in the product it "
                           "comes from and has tier 1 or 2"})
    enc = {k: {"zlib": True, "complevel": 4} for k in ds.data_vars}
    ds.to_netcdf(f"{cfg.file_daily}.tmp", encoding=enc)
    os.replace(f"{cfg.file_daily}.tmp", cfg.file_daily)
    logging.info("wrote %s", cfg.file_daily)
    return {"prec": prec, "max_1h": max1h}


def write_stations(codes, names, lat, lon, alt, hym, bal, arn, t10, t1h, p1h, daily):
    """One row per station: metadata, sources and how much usable data it has."""
    vh = np.isfinite(p1h.reshape(len(codes), -1, 6)).all(2)
    arn_codes = set(arn.INDICATIVO)
    has = np.isfinite(daily["prec"])
    first = [str(DAYS[np.argmax(r)].date()) if r.any() else "" for r in has]
    last = [str(DAYS[len(r) - 1 - np.argmax(r[::-1])].date()) if r.any() else "" for r in has]
    df = pd.DataFrame({
        "code": codes, "name": names, "lat": np.round(lat, 5), "lon": np.round(lon, 5),
        "alt": alt, "in_hymex": [c in hym for c in codes], "in_balear": [c in bal for c in codes],
        "in_arnau": [c in arn_codes for c in codes],
        "frac_valid_hours": np.round(vh.mean(1), 3),
        "frac_days_verified_1h": np.round((t1h == cfg.TIER_A).mean(1), 3),
        "frac_days_complete_daily": np.round(has.mean(1), 3),
        "first_complete_day": first, "last_complete_day": last})
    df.to_csv(cfg.file_stations, index=False)
    logging.info("wrote %s", cfg.file_stations)


def write_readme(lines, codes):
    """README for the database directory, with the numbers of this build."""
    text = f"""# AEMET combined rain-gauge database, {cfg.syear}-{cfg.eyear}

Built by `make_station_dataset.py` (MCS-tracking/Observations) on
{pd.Timestamp.now():%Y-%m-%d %H:%M}. Do not edit by hand: re-run the script.
{len(codes)} stations with 10-min data. All times UTC.

**Restricted data:** the HyMEX 10-min records may only be used within HyMeX
projects, and the files here inherit that restriction.

## Files

| File | Content |
|---|---|
| `AEMET_10MIN_PREC_{cfg.syear}-{cfg.eyear}.nc` | `prec(station, time)`, mm per 10 min, stamped at the START of the interval; `day_tier`, `day_source` |
| `AEMET_01H_PREC_{cfg.syear}-{cfg.eyear}.nc` | `prec(station, time)`, mm per clock hour, stamped at the START of the hour; hourly rules (below) |
| `AEMET_DAILY_{cfg.syear}-{cfg.eyear}.nc` | per day: total, max 10-min, max sliding 60-min, max clock hour, times of the maxima, valid counts, tiers (Arnau-style) |
| `AEMET_ARNAU_DAILY_{cfg.syear}-2019.nc` | AEMET climatological daily record (P24, PMAX10..PMAX12H, flags) for 403 stations: the validated reference |
| `stations.csv` | code, name, position, sources, fraction of valid hours and of verified days |
| `build_summary.txt` | tier statistics of this build |
| `arnau_raw/` | the Arnau archives, unpacked |

## Sources

- **HyMEX 10-min:** AEMET real-time AWS database, not quality controlled.
- **Balearic 10-min:** AEMET historical archive, cuenca B.
- **Arnau:** AEMET climatological daily database, with quality flags. It is
  mostly derived from the same 10-min records, which is what makes it usable
  as their quality control.

## Quality control: day tiers

| Tier | Meaning |
|---|---|
| 1 verified | reproduces Arnau's validated daily total and max 60-min amount (and max 10-min in the 10-min product), within {cfg.tol_abs} mm + {cfg.tol_rel:.0%} |
| 2 automatic | no usable Arnau record that day; passes the automatic checks (10-min <= {cfg.max_10min} mm, day <= {cfg.max_day} mm, no stuck sensor, no isolated spike) |
| 8 rejected | disagrees with Arnau (when Arnau is derived from the same record), or Arnau flags the day doubtful |
| 9 rejected | fails an automatic check |

Only tiers 1 and 2 carry data.

- **10-min rules** also require the maximum 10-min amount to match Arnau.
- **Hourly rules** do not use the 10-min maximum: a 10-min timing difference
  inside an hour does not make the hourly totals wrong.
- **In verified days, missing 10-min steps are 0** in the 1-h product and in
  the daily maxima. The day reproduces AEMET's validated total, so the gaps
  held no rain. This is where most of the hourly gain comes from.
- **Where both 10-min sources have a day,** the Balearic archive is preferred
  if it passes; on days both verify, the two are identical.

## Conventions

- **Source stamps.** Both 10-min sources stamp the interval end; this was
  measured from exact daily matches with Arnau, not assumed. The files here
  are stamped at the start.
- **Station codes.** HyMEX synoptic codes (08xxx) are mapped to climatological
  codes (INDICATIVO).
- **Daily values:** days run 00-24 UTC. A daily value is present only if its
  day is complete in the product it comes from.

## Build summary

```
""" + "\n".join(lines) + "\n```\n"
    with open(f"{cfg.path_db}/README.md", "w") as fh:
        fh.write(text)


def write_arnau(arn, meta):
    """Arnau daily record, all its stations, for daily and extreme-value use."""
    codes = sorted(c for c in arn.INDICATIVO.unique() if c in meta)
    days = pd.date_range(f"{cfg.syear}-01-01", arn.date.max(), freq="D")
    fields = ["P24", "PMAX10", "PMAX20", "PMAX30", "PMAX60", "PMAX2H", "PMAX6H", "PMAX12H"]
    arn = arn[arn.INDICATIVO.isin(codes)].set_index(["INDICATIVO", "date"])
    full = pd.MultiIndex.from_product([codes, days], names=["INDICATIVO", "date"])
    arn = arn.reindex(full)
    data = {}
    for k in fields:
        raw = arn[k].values.astype(float)
        val = np.where(raw == -3, 0.0, np.where(raw < 0, np.nan, raw / 10.0))
        data[k.lower()] = (("station", "day"), val.reshape(len(codes), days.size).astype("float32"),
                           {"units": "mm", "comment": "tenths converted to mm; -3 (trace) -> 0; "
                                                      "-4 (accumulated) -> NaN"})
    for k in ("ID_FLAG_Q1", "ID_FLAG_Q2", "ID_FLAG_E", "ID_FLAG_P"):
        v = arn[k].values
        data[k.lower()] = (("station", "day"),
                           np.where(np.isfinite(v), v, -1).reshape(len(codes), days.size).astype("i1"))
    lat = np.array([meta[c][0] for c in codes])
    lon = np.array([meta[c][1] for c in codes])
    alt = np.array([meta[c][2] for c in codes])
    names = np.array([meta[c][3] for c in codes], dtype=object)
    ds = xr.Dataset(data, coords={"day": days, **station_coords(codes, names, lat, lon, alt)},
                    attrs={"title": "AEMET daily precipitation and sub-daily maxima (Arnau)",
                           "day_convention": "00-24 UTC",
                           "maxima": "running windows within the day at 10-min steps",
                           "flags": "Q: 0 manual ok, 1 auto ok, 10 unchecked, 20/21 doubtful; "
                                    "E: 0 original, 10 modified; P: 0 not from 10-min, "
                                    "1 fully, 2 partly from 10-min; -1 missing"})
    enc = {v: {"zlib": True} for v in ds.data_vars}
    ds.to_netcdf(f"{cfg.file_arnau}.tmp", encoding=enc)
    os.replace(f"{cfg.file_arnau}.tmp", cfg.file_arnau)
    logging.info("wrote %s (%d stations)", cfg.file_arnau, len(codes))


if __name__ == "__main__":
    main()
