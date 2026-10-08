"""Learn habitat weights, rain/temperature timing and season from public records.

Run in GitHub Actions (`.github/workflows/calibrate.yml`, monthly or by hand):
    python -m pipeline.calibrate --cache cache/static --out calibration

Method
------
* Records: GBIF finds per species and region (incl. iNaturalist research grade).
* Background ("target group"): all fungi records on GBIF in the same region, i.e. places
  where mushroom recorders actually go. Comparing finds to these, instead of random
  ground, cancels most of the bias towards paths, towns and popular forests.
* Habitat: regularised logistic regression on per-cell features (tree types, grass types,
  pH, altitude, slope, dampness, satellite wetness) plus the hand-set rule score. Tested
  with 5-fold spatial-block cross-validation; used only where it beats the rules.
* Timing: daily historical weather (Open-Meteo archive, 0.5 deg cells) for every dated
  find since 2020 and for nearby "other" days at the same place. A grid search picks the
  rain delay window, rain needed and temperature range that best separate find days from
  other days. Trained on earlier years, tested on the last two.
* Season: each species' share of all fungi records by month (corrects for when people
  go out recording), smoothed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import os
import time

import numpy as np
import requests
from scipy.stats import rankdata

from . import features, habitat as hab_mod, log, run as R, satellite, scoring, sightings, static_layers
from .grid import Grid

GBIF = "https://api.gbif.org/v1"
FUNGI_KEY = 5
ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
HIST_START = "2020-01-01"
WINDOW = 45           # days of rain history kept per point
MIN_HAB = 40          # finds needed to fit habitat
MIN_TIMING = 50       # dated finds needed to fit timing
MIN_SEASON = 50       # finds needed to learn season
MIN_SKILL = 0.60      # a learned habitat model must reach at least this AUC to be used
MIN_TIMING_SKILL = 0.58  # ... and learned timing at least this (test years)


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 3 or len(neg) < 3:
        return None
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ============================================================== GBIF helpers
def gbif_background(bbox, per_year=300):
    w, s, e, n = bbox
    out, this_year = [], dt.date.today().year
    for y in range(2008, this_year + 1):
        try:
            r = requests.get(f"{GBIF}/occurrence/search", timeout=90, params={
                "taxonKey": FUNGI_KEY, "hasCoordinate": "true", "hasGeospatialIssue": "false",
                "occurrenceStatus": "PRESENT", "decimalLatitude": f"{s},{n}", "decimalLongitude": f"{w},{e}",
                "year": y, "limit": per_year})
            r.raise_for_status()
            for o in r.json().get("results", []):
                unc = o.get("coordinateUncertaintyInMeters")
                if unc is not None and unc > 1000:
                    continue
                out.append({"lat": o["decimalLatitude"], "lon": o["decimalLongitude"],
                            "date": (o.get("eventDate") or "")[:10], "month": o.get("month")})
        except Exception as exc:
            log.warn(f"GBIF background {y}: {exc.__class__.__name__}")
        time.sleep(0.3)
    return out


def month_counts(bbox, taxon_keys):
    w, s, e, n = bbox
    r = requests.get(f"{GBIF}/occurrence/search", timeout=90, params={
        "taxonKey": taxon_keys, "hasCoordinate": "true", "decimalLatitude": f"{s},{n}",
        "decimalLongitude": f"{w},{e}", "facet": "month", "facetLimit": 12, "limit": 0})
    r.raise_for_status()
    counts = np.zeros(12)
    for f in r.json().get("facets", []):
        for c in f.get("counts", []):
            m = int(c["name"])
            if 1 <= m <= 12:
                counts[m - 1] += c["count"]
    return counts


def species_keys(sp):
    keys = []
    for name in sp.get("gbif", []):
        try:
            k = sightings._taxon_key(name)
            if k:
                keys.append(k)
        except Exception:
            pass
    return keys


# ============================================================== habitat
def _cells(points, grid: Grid):
    rows, cols, keep = [], [], []
    for i, p in enumerate(points):
        r, c = grid.lonlat_to_cell(p["lon"], p["lat"])
        r, c = int(r), int(c)
        if 0 <= r < grid.height and 0 <= c < grid.width:
            rows.append(r)
            cols.append(c)
            keep.append(i)
    return np.array(rows, int), np.array(cols, int), keep


def fit_habitat(region, species, cache_dir):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    rid = region["id"]
    grid = Grid.from_bbox(region["bbox"], region.get("zoom", 10))
    layers = static_layers.load_all(region, grid, cache_dir)
    terr = hab_mod.terrain(layers["elev"], grid)
    sat_wet = satellite.relative_wetness(layers["ndmi"], layers) if "ndmi" in layers else None
    shared = features.smooth(features.shared_features(layers, terr, sat_wet, grid.shape))
    names = sorted(shared)
    land = layers.get("lc_water")
    land = (land.astype(np.float32) < 0.5) if land is not None else np.ones(grid.shape, bool)

    recs = R.load_sightings(region, species, cache_dir)
    bg = gbif_background(region["bbox"])
    br, bc, _ = _cells(bg, grid)
    log.info(f"{rid}: {len(br)} target-group background records (all fungi)")
    out = {"background_n": int(len(br))}
    for sp in species:
        sid = sp["id"]
        rule = features.smooth({"r": hab_mod.habitat(sp, layers, grid, terr)})["r"]
        mask = features.smooth({"m": features.host_mask(sp, layers, grid.shape)})["m"]
        pr, pc, _ = _cells(recs.get(sid, []), grid)
        res = {"n": int(len(pr)), "use": False}
        if len(pr) < 5 or len(br) < 50:
            out[sid] = res
            continue
        res["auc_rule"] = auc(rule[pr, pc], rule[br, bc])
        if len(pr) < MIN_HAB:
            out[sid] = res
            continue
        cols_ = ["rule"] + names
        def X_at(r, c):
            return np.column_stack([rule[r, c]] + [shared[k][r, c] for k in names])
        X = np.vstack([X_at(pr, pc), X_at(br, bc)]).astype(np.float64)
        y = np.r_[np.ones(len(pr)), np.zeros(len(br))]
        mean, std = X.mean(0), X.std(0)
        std[std < 1e-6] = 1.0
        Z = (X - mean) / std
        rows = np.r_[pr, br]
        cols = np.r_[pc, bc]
        groups = (rows // 150) * 10000 + (cols // 150)       # ~16-25 km spatial blocks
        oof = np.zeros(len(y))
        n_groups = len(np.unique(groups))
        splitter = (GroupKFold(n_splits=5).split(Z, y, groups) if n_groups >= 5
                    else StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(Z, y))
        if True:
            for tr, te in splitter:
                if y[tr].sum() < 5 or y[te].sum() < 1:
                    oof[te] = Z[te, 0] - 10
                    continue
                m = LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000).fit(Z[tr], y[tr])
                oof[te] = m.decision_function(Z[te])
            # the learned score is always limited to cells with some host (trees / unploughed grass)
            oof = 1 / (1 + np.exp(-oof)) * mask[rows, cols]
            res["auc_fit_cv"] = auc(oof[y == 1], oof[y == 0])
        m = LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000).fit(Z, y)
        model = {"features": cols_, "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0]),
                 "mean": mean.tolist(), "std": std.tolist(), "scale": 1.0}
        raw = features.predict(model, rule, shared) * mask
        model["scale"] = float(np.quantile(raw[land], 0.995)) if land.any() else 1.0
        res["model"] = model
        order = np.argsort(-np.abs(m.coef_[0]))[:6]
        res["top_factors"] = [{"feature": cols_[i], "weight": round(float(m.coef_[0][i]), 3)} for i in order]
        res["use"] = bool(res.get("auc_fit_cv") is not None and res.get("auc_rule") is not None
                          and res["auc_fit_cv"] >= max(res["auc_rule"] + 0.02, MIN_SKILL))
        log.info(f"{rid} {sid}: n={len(pr)} rule AUC {res['auc_rule']:.3f} fitted (CV) "
                 f"{res.get('auc_fit_cv') or float('nan'):.3f} -> {'LEARNED' if res['use'] else 'rules kept'}; "
                 f"top: {[(t['feature'], t['weight']) for t in res['top_factors']]}")
        out[sid] = res
    return out, recs


# ============================================================== weather history
def _cell_key(lat, lon):
    return f"{round(lat * 2) / 2:.1f}_{round(lon * 2) / 2:.1f}"


def load_history(keys, hist_dir, end):
    os.makedirs(hist_dir, exist_ok=True)
    have, need = {}, []
    for k in keys:
        p = os.path.join(hist_dir, f"c_{k}.npz")
        if os.path.exists(p):
            with np.load(p) as z:
                if str(z["end"]) >= (dt.date.fromisoformat(end) - dt.timedelta(days=40)).isoformat():
                    have[k] = {"start": str(z["start"]), "P": z["P"], "T": z["T"], "Tmin": z["Tmin"]}
                    continue
        need.append(k)
    log.info(f"weather history: {len(have)} cells cached, fetching {len(need)}")
    for i in range(0, len(need), 6):
        batch = need[i:i + 6]
        lats = [float(k.split("_")[0]) for k in batch]
        lons = [float(k.split("_")[1]) for k in batch]
        data = None
        for attempt in range(5):
            try:
                r = requests.get(ARCHIVE_API, timeout=(20, 180), params={
                    "latitude": ",".join(map(str, lats)), "longitude": ",".join(map(str, lons)),
                    "start_date": HIST_START, "end_date": end, "timezone": "UTC",
                    "daily": "precipitation_sum,temperature_2m_mean,temperature_2m_min"})
                if r.status_code == 200:
                    data = r.json()
                    break
                wait = 70 if r.status_code == 429 else 20
                log.warn(f"history HTTP {r.status_code}: {r.text[:100]}; retry in {wait}s")
            except requests.RequestException as exc:
                wait = 20
                log.warn(f"history {exc.__class__.__name__}; retry in {wait}s")
            time.sleep(wait)
        if data is None:
            continue
        data = data if isinstance(data, list) else [data]
        for k, d in zip(batch, data):
            dd = d["daily"]
            arr = lambda v: np.array([np.nan if x is None else x for x in dd[v]], np.float32)
            rec = {"start": dd["time"][0], "P": np.nan_to_num(arr("precipitation_sum")),
                   "T": arr("temperature_2m_mean"), "Tmin": arr("temperature_2m_min")}
            np.savez_compressed(os.path.join(hist_dir, f"c_{k}.npz"), end=dd["time"][-1], **rec)
            have[k] = rec
        time.sleep(12)
    return have


# ============================================================== timing
LAG0 = [3, 5, 7, 10, 14]
WIDTH = [5, 8, 12, 16]
NEED = [10, 20, 35, 50]
TCENTER = [4, 6, 8, 10, 12, 14, 16, 18]
THALF = [3, 5, 7]


def _point_windows(points, hist, rng, n_bg=4):
    """Return arrays for find days (y=1) and nearby other days (y=0)."""
    Pw, T7, FD, Y, YR = [], [], [], [], []
    for p in points:
        h = hist.get(_cell_key(p["lat"], p["lon"]))
        if h is None or len(p.get("date", "")) < 10:
            continue
        d0 = dt.date.fromisoformat(h["start"])
        try:
            d = (dt.date.fromisoformat(p["date"]) - d0).days
        except ValueError:
            continue
        offs = [0] + list(rng.choice([-1, 1], n_bg) * rng.integers(10, 46, n_bg))
        for j, o in enumerate(offs):
            t = d + int(o)
            if t < WINDOW or t >= len(h["P"]):
                continue
            Pw.append(h["P"][t - WINDOW + 1:t + 1][::-1])
            T7.append(np.nanmean(h["T"][t - 6:t + 1]))
            FD.append(float((h["Tmin"][t - 3:t + 1] < -1.5).sum()))
            Y.append(1 if j == 0 else 0)
            YR.append(int(p["date"][:4]))
    if not Y:
        return None
    return (np.array(Pw, np.float32), np.array(T7, np.float32), np.array(FD, np.float32),
            np.array(Y), np.array(YR))


def _weather_score(Pw, T7, FD, lag, need, temp, frost):
    w = scoring.lag_weights(lag[0], lag[1])[:WINDOW]
    w = np.pad(w, (0, WINDOW - len(w)))
    rain = Pw @ w
    s = np.clip(rain / need, 0, 1) * hab_mod.trapezoid(T7, *temp)
    return s * (1 - frost * np.clip(FD / 3, 0, 1))


def fit_timing(sp, points, hist, rng):
    arr = _point_windows(points, hist, rng)
    w = sp["weather"]
    if arr is None:
        return {"n": 0, "use": False}
    Pw, T7, FD, Y, YR = arr
    last = int(YR.max())
    test = YR >= last - 1
    train = ~test
    res = {"n": int(Y.sum()), "n_train": int(Y[train].sum()), "n_test": int(Y[test].sum()), "use": False}
    rule = _weather_score(Pw, T7, FD, w["lag"], w["rain_need"], w["temp"], w["frost"])
    res["auc_rule_train"] = auc(rule[train & (Y == 1)], rule[train & (Y == 0)])
    res["auc_rule_test"] = auc(rule[test & (Y == 1)], rule[test & (Y == 0)])
    if res["n_train"] < MIN_TIMING:
        return res
    # precompute rain sums per lag window to keep the search fast
    best = None
    for l0, wd in itertools.product(LAG0, WIDTH):
        lw = scoring.lag_weights(l0, l0 + wd)[:WINDOW]
        rain = Pw @ np.pad(lw, (0, WINDOW - len(lw)))
        for need, c, hw in itertools.product(NEED, TCENTER, THALF):
            temp = [c - hw - 3, c - hw, c + hw, c + hw + 3]
            s = np.clip(rain / need, 0, 1) * hab_mod.trapezoid(T7, *temp) * (1 - w["frost"] * np.clip(FD / 3, 0, 1))
            a = auc(s[train & (Y == 1)], s[train & (Y == 0)])
            if a is not None and (best is None or a > best[0]):
                best = (a, [l0, l0 + wd], need, temp)
    a, lag, need, temp = best
    fit = _weather_score(Pw, T7, FD, lag, need, temp, w["frost"])
    res.update({"auc_fit_train": a, "auc_fit_test": auc(fit[test & (Y == 1)], fit[test & (Y == 0)]),
                "params": {"lag": lag, "rain_need": need, "temp": temp},
                "rule_params": {"lag": w["lag"], "rain_need": w["rain_need"], "temp": w["temp"]}})
    if res["n_test"] >= 20 and res["auc_fit_test"] is not None and res["auc_rule_test"] is not None:
        res["use"] = bool(res["auc_fit_test"] >= max(res["auc_rule_test"] + 0.02, MIN_TIMING_SKILL))
    elif res["n_train"] >= 100 and res["auc_rule_train"] is not None:
        res["use"] = bool(a >= max(res["auc_rule_train"] + 0.04, MIN_TIMING_SKILL + 0.04))
    return res


def season_weights(counts_sp, counts_all):
    ratio = counts_sp / (counts_all + 1.0)            # species' share of fungi records each month
    sm = 0.5 * ratio + 0.25 * np.roll(ratio, 1) + 0.25 * np.roll(ratio, -1)
    wts = (sm / max(sm.max(), 1e-9)) ** 1.5             # sharpen a little: off-season months -> ~0
    return np.clip(np.round(wts, 2), 0.02, 1).tolist()


# ============================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(R.ROOT, "cache", "static"))
    ap.add_argument("--out", default=os.path.join(R.ROOT, "calibration"))
    ap.add_argument("--history", default=os.path.join(R.ARCHIVE_DIR, "history"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    regions = R.load_yaml("regions.yaml")["regions"]
    species = R.load_yaml("species.yaml")["species"]
    rng = np.random.default_rng(7)
    cal = {"generated": dt.datetime.utcnow().isoformat(timespec="minutes") + "Z",
           "habitat": {}, "timing": {}, "season": {}}

    by_country = {}
    for region in regions:
        try:
            h, recs = fit_habitat(region, species, args.cache)
            cal["habitat"][region["id"]] = h
            by_country.setdefault(region.get("country", "XX"), []).append((region, recs))
        except Exception as exc:
            log.error(f"habitat calibration failed for {region['id']}", exc)

    end = (dt.date.today() - dt.timedelta(days=7)).isoformat()
    keys_needed = set()
    for country, items in by_country.items():
        for region, recs in items:
            for sp in species:
                for p in recs.get(sp["id"], []):
                    if p.get("date", "")[:4] >= "2020":
                        keys_needed.add(_cell_key(p["lat"], p["lon"]))
    hist = load_history(sorted(keys_needed), args.history, end)

    for country, items in by_country.items():
        cal["timing"][country] = {}
        cal["season"][country] = {}
        try:
            fungi = sum(month_counts(r["bbox"], [FUNGI_KEY]) for r, _ in items)
        except Exception as exc:
            log.warn(f"fungi month counts failed: {exc}")
            fungi = None
        for sp in species:
            pts = [p for _, recs in items for p in recs.get(sp["id"], []) if p.get("date", "")[:4] >= "2020"]
            try:
                t = fit_timing(sp, pts, hist, rng)
            except Exception as exc:
                log.error(f"timing fit failed {country} {sp['id']}", exc)
                t = {"use": False}
            cal["timing"][country][sp["id"]] = t
            log.info(f"timing {country} {sp['id']}: n={t.get('n')} rule test AUC {t.get('auc_rule_test')} "
                     f"fitted test {t.get('auc_fit_test')} params {t.get('params')} -> "
                     f"{'LEARNED' if t.get('use') else 'rules kept'}")
            try:
                keys = species_keys(sp)
                cs = sum(month_counts(r["bbox"], keys) for r, _ in items) if keys else np.zeros(12)
                s = {"n": int(cs.sum()), "counts": cs.astype(int).tolist(), "use": False}
                if fungi is not None and cs.sum() >= MIN_SEASON:
                    s["weights"] = season_weights(cs, fungi)
                    s["use"] = True
                s["rule_weights"] = sp["weather"]["season"]
                cal["season"][country][sp["id"]] = s
            except Exception as exc:
                log.warn(f"season {country} {sp['id']}: {exc}")

    with open(os.path.join(args.out, "calibration.json"), "w") as f:
        json.dump(cal, f, indent=1)
    with open(os.path.join(args.out, "REPORT.md"), "w") as f:
        f.write(report(cal, species))
    with open(os.path.join(args.out, "log.json"), "w") as f:
        json.dump(log.ENTRIES, f, indent=1, ensure_ascii=False)
    log.info("calibration written")


def report(cal, species):
    names = {s["id"]: s["name"] for s in species}
    L = [f"# Model calibration ({cal['generated']})", "",
         "AUC = how often a real find scores above a comparison point (0.5 = no skill, 1 = perfect).",
         "Habitat: finds vs all fungi records in the region (spatial cross-validation).",
         "Timing: find days vs nearby other days at the same place (trained on earlier years, tested on the last two).", ""]
    L += ["## Habitat", "", "| Region | Species | Finds | Rules | Learned (CV) | Used |", "|---|---|---|---|---|---|"]
    for rid, d in cal["habitat"].items():
        for sid, r in d.items():
            if sid == "background_n":
                continue
            L.append(f"| {rid} | {names.get(sid, sid)} | {r.get('n')} | {_f(r.get('auc_rule'))} | "
                     f"{_f(r.get('auc_fit_cv'))} | {'yes' if r.get('use') else 'no'} |")
    L += ["", "## Timing", "", "| Country | Species | Dated finds | Rules (test) | Learned (test) | Learned params | Used |",
          "|---|---|---|---|---|---|---|"]
    for c, d in cal["timing"].items():
        for sid, r in d.items():
            L.append(f"| {c} | {names.get(sid, sid)} | {r.get('n')} | {_f(r.get('auc_rule_test'))} | "
                     f"{_f(r.get('auc_fit_test'))} | {r.get('params', '')} | {'yes' if r.get('use') else 'no'} |")
    L += ["", "## Season (month weights Jan..Dec)", ""]
    for c, d in cal["season"].items():
        for sid, r in d.items():
            if r.get("use"):
                L.append(f"- {c} {names.get(sid, sid)} (n={r['n']}): {r['weights']}")
    return "\n".join(L) + "\n"


def _f(x):
    return "-" if x is None else f"{x:.2f}"


def _main_logged():
    out = os.path.join(R.ROOT, "calibration")
    for i, a in enumerate(os.sys.argv):
        if a == "--out" and i + 1 < len(os.sys.argv):
            out = os.sys.argv[i + 1]
    try:
        main()
    except Exception as exc:
        log.error("calibration failed", exc)
        raise
    finally:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "log.json"), "w") as f:
            json.dump(log.ENTRIES, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    _main_logged()
