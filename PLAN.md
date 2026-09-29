# Work plan — Mediterranean convective storms (EPICC)

Status as of **2026-09-29**. Companion to `Observations/README.md`, which
documents how the pipeline works; this file is what is left to do.

---

## Stage 0 — timestamp fix (RESOLVED 2026-09-29)

**What was wrong.** EPICC 10-minute rain is WRF `PREC_ACC_NC` (rain over the
*last* 10 min, end-stamped), and `cdo hoursum` grouped stamps HH:00..HH:50, so
the original `UIB_01H_RAIN` files cover **HH-1:50 to HH:50** — ten minutes early.
Verified against `RAINNC` in `wrfout_d01_2020-01-10`: summing stamps
HH:10..HH+1:00 matches to 3e-6 mm, HH:00..HH:50 does not.

**Corrected files** are in `/scratch3/dargueso/postprocessed/EPICC/<run>/RAIN_clockhour/`
(120 months per run, both present), built by
`EPICC_scripts/WRF_processing/fix_hourly_rain_clockhour.py`. Originals kept —
they are used by another manuscript.

| file | stamp | window |
|---|---|---|
| `RAIN/` (original) | HH:25 | 00:00–00:50, i.e. HH-1:50 to HH:50 — **10 min early** |
| `RAIN_clockhour/` | **HH:30** | **00:00–01:00** — correct |
| `OLR/` | HH:00 | instantaneous |
| `WSPD10/`, `PW/`, `CAPE2D/` | HH:00 | instantaneous |

### A silent bug this introduced, already fixed

A storm stamped **HH:30** is *exactly equidistant* from the hourly fields at
HH:00 and HH+1:00, and pandas breaks that tie by rounding **up**. Verified:

| storm stamp | `method='nearest'` picked |
|---|---|
| 00:25 (original files) | 00:00 — correct |
| **00:30 (clock-hour files)** | **01:00 — the wrong hour** |

Every storm would have been paired with wind, PW and CAPE from the *following*
hour: a systematic one-hour lag across all environmental fields, with no error
and entirely plausible output. `plot_scatter_hist_storm_characteristics.py` and
`..._multiple_exp.py` now floor the storm time to the clock hour and select
exactly, which is correct under either convention.

### Still to check before re-running

- **Point the tracker at `RAIN_clockhour/`.** `MCS_tracking_WRF.py` globs
  `{path_in}/RAIN/UIB_01H_RAIN_*.nc`; that is still the uncorrected directory.
- **OLR is unaffected** (instantaneous, not an accumulation) but note the
  relative alignment shifts: RAIN element *i* now covers 00:00–01:00 while OLR
  element *i* is the instantaneous field at 00:00, so the cloud field now sits at
  the *start* of the rain window rather than ~25 min into it. Defensible either
  way, worth one sentence in the Methods.
- **`Observations/make_model_tracking_input.py`** floors model stamps to the
  hour; HH:30 still floors to HH:00, so it stays correct — but it must read the
  corrected files too.
- The aggregation fix is **uncommitted** in both `EPICC_scripts` and
  `wrfprocessing` (`create_*_freq_files`, `compute_RAIN` guard).

### Everything tracked so far is stale — discard and re-run

| run | set | state |
|---|---|---|
| ERA5 | `ConvStormTracking` (scratch1) | 120/120 — original SB, also predates the area-weighted volume |
| ERA5 | `ConvStormTracking_SB` (scratch3) | 120/120 — completed just before cancellation |
| ERA5 | `ConvStormTracking_YS` (scratch3) | 119/120 |
| CMIP6anom | `ConvStormTracking` (scratch1) | 120/120 — original SB |
| CMIP6anom | `ConvStormTracking_SB` (scratch3) | 0/120 — cancelled at start |
| CMIP6anom | `ConvStormTracking_YS` (scratch3) | 120/120 |

## Stage 1 — re-run tracking (~5 h)

Both conversions, both runs, 2011–2020, `exp1`. `run_sb_then_summaries.sh` does
SB tracking then summaries; the YS equivalent is `run_ys_tracking.sh`. Both just
need re-launching once the inputs are fixed.

- Watch for **`2020-02` present-day**, which produced zero MCS objects under YS.
  That was genuine (0 MCS against 3,528 precipitation and 30,638 cloud objects,
  no errors), but if it recurs after the fix it deserves a second look rather
  than being assumed.
- A month with no qualifying storm writes **no `MCS_` pickle** — the month count
  being under 120 is expected, not a failure.

## Stage 2 — summaries (~1–2 h)

ASON 2011–2020, WME, `exp1`, for both SB and YS. One run with
`calc_summary=True` builds the summaries *and* draws the main figure, so both
fall out together. Use the `py310` environment — it is the only one with both
`wrf` and `dask`.

Set `calc_summary=False` afterwards, and make sure the other five plotting
scripts have matching `syear` / `allmonths`, or they will look for a summary
filename that does not exist (the guard will say so).

## Stage 3 — redo the two checks that the re-run invalidates

1. **Brightness-temperature robustness** — `Plotting/check_bt_robustness.py`.
   Will be fully like-for-like this time (both sets tracked by current code, both
   with area-weighted volume), so the ~7% volume caveat can come out of the
   manuscript note. Previous result, now provisional:

   | metric | SB | YS | overlap |
   |---|---|---|---|
   | count | 0.70 [0.66, 0.75] | 0.64 [0.57, 0.71] | yes |
   | area | 1.34 [1.28, 1.41] | 1.39 [1.33, 1.52] | yes |
   | duration | 1.08 [1.00, 1.17] | 1.15 [1.00, 1.21] | yes |
   | peak rain | 1.48 [1.42, 1.53] | 1.49 [1.45, 1.54] | yes |
   | volume | 1.65 [1.53, 1.80] | 1.76 [1.62, 1.93] | yes |

2. **Observational comparison.** The observational tracking itself is
   unaffected — IMERG and MERGIR are independent of the WRF postprocessing — but
   `mod0.1_*` was coarsened from the same RAIN/OLR files, so
   `make_model_tracking_input.py` and everything downstream of it must re-run.

---

## Stage 4 — analyses

### Essential for the manuscript

- **Frequency change** (the headline). Currently −30% (SB) / −36% (YS) for ASON
  2011–2020, against the draft's provisional −41% to −45%, which came from a
  different period and a span of configurations. Needs settling with final
  numbers.
- **Volume decomposition** — Δln R = Δln N + Δln A + Δln I + Δln D + ε. The
  Methods describe it; nothing has computed it. With count 0.64, area 1.39, peak
  1.49 and duration 1.15 the terms should combine into roughly the 1.76 volume
  ratio — worth verifying they do, since a large residual would mean the
  decomposition is hiding covariance.
- **Wind significance** — the draft's oldest pending item ("less clear" vs a
  visible shift). Now computable from the WSPD10 summaries. Note the definition
  changed: it is the maximum of *hourly-mean* wind speed over the footprint, not
  a gust.

### Valuable, and cheap given what already exists

- **Sensitivity across exp1–exp5 under YS.** The draft claims the decrease is
  robust across tracker configurations; that is currently supported by SB
  evidence only.
- **Storm-relative structure at 2 km.** `Observations/plot_storm_structure.py`
  works on any RAIN + tracking pair. At 0.1° it found the mechanism behind the
  area deficit (core right, shield too narrow). Applied to present vs PGW it
  would show whether the projected change is core intensification or shield
  expansion — a genuinely new result rather than an evaluation.
- **Change maps.** `Observations/plot_obs_model_maps_relative.py` applied to
  pres vs fut instead of model vs obs, answering "where do the extra large
  storms appear".

### Worth considering

- **Extending beyond ASON.** All twelve months are tracked. The caveat is the
  winter deficit found in the evaluation: several Dec–Mar months contain no
  qualifying storm at all in the model against ~10 observed.
- **The coarsened climate signal.** `mod0.1_*_fut` is tracked but unused.
  Comparing the PGW change at 0.1° against the 2 km change is the actual test of
  "if they agree statistically, use 2 km only".

---

## Standing items, independent of the timestamp fix

- **The manuscript still says ASO in 10 places** while the analysis is ASON —
  Results, the scaling and decomposition paragraphs, and the Fig. 1 and Fig. 2
  captions. Every ASO-based number also needs recomputing over four months.
- **Gauge evaluation** found the model 10–30% too intense, with a manuscript
  claim flagged; EURADCLIM run pending after download.
- **Supplementary Fig. 1** numbers are in the draft's author-notes table but the
  figure itself is not yet placed in the supplement.
- **Figure format** — figures are 200 dpi PNG; NCC will want vector for final
  submission.

## Decisions already made (do not relitigate)

- **Yang & Slingo, not Stefan-Boltzmann**, for OLR → brightness temperature at
  both 2 km and 0.1°. The 241 K / 225 K thresholds come from the satellite MCS
  literature and are defined on window-channel Tb.
- **`exp` means thresholds and nothing else.** Anything else that changes the
  result names the dataset or directory. Never `exp1b`.
- **`WSPD10`, not `U10MET`/`V10MET`**, for storm wind.
- **Model coarsened to the observational 0.1° grid** for the evaluation, not
  compared across resolutions.
