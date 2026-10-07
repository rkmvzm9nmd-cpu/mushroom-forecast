"""Daily build: static layers (cached) -> habitat -> weather -> scores -> site data -> alerts.

Usage:  python -m pipeline.run [--out build/site] [--cache cache/static]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import time
from zoneinfo import ZoneInfo

import numpy as np
import yaml

from . import archive, alerts, habitat as hab_mod, log, render, scoring, sightings, static_layers, weather
from .grid import Grid, downsample, upsample

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOOD = 0.4          # score counted as "good" in stats and alerts
HALF = 2            # weather is modelled at half resolution then smoothed up
GBIF_MAX_AGE_DAYS = 7
ARCHIVE_DIR = os.environ.get("WEATHER_ARCHIVE", os.path.join(ROOT, "archive"))


def load_yaml(name):
    with open(os.path.join(ROOT, "config", name), encoding="utf-8") as f:
        return yaml.safe_load(f)


def site_url():
    repo = os.environ.get("GITHUB_REPOSITORY", "rkmvzm9nmd-cpu/mushroom-forecast")
    owner, name = repo.split("/")
    return f"https://{owner.lower()}.github.io/{name}/"


def load_sightings(region, species, cache_dir):
    path = os.path.join(cache_dir, f"{region['id']}_gbif.json")
    if os.path.exists(path):
        with open(path) as f:
            cached = json.load(f)
        age = (time.time() - cached.get("fetched", 0)) / 86400
        if age < GBIF_MAX_AGE_DAYS and set(cached.get("records", {})) == {s["id"] for s in species}:
            return cached["records"]
    try:
        recs = sightings.fetch(species, region["bbox"])
        with open(path, "w") as f:
            json.dump({"fetched": time.time(), "records": recs}, f)
        return recs
    except Exception as exc:
        log.error("GBIF fetch failed", exc)
        return {}


def process_region(region, species, out_dir, cache_dir):
    rid = region["id"]
    grid = Grid.from_bbox(region["bbox"], region.get("zoom", 10))
    log.info(f"{rid}: grid {grid.width}x{grid.height} px at zoom {grid.zoom}")
    rdir = os.path.join(out_dir, "data", rid)
    for sub in ("habitat", "score", "wx", "rain"):
        os.makedirs(os.path.join(rdir, sub), exist_ok=True)

    layers = static_layers.load_all(region, grid, cache_dir)
    terr = hab_mod.terrain(layers["elev"], grid)
    H = {}
    for sp in species:
        H[sp["id"]] = hab_mod.habitat(sp, layers, grid, terr)
        render.save_png(H[sp["id"]], os.path.join(rdir, "habitat", f"{sp['id']}.png"))

    # --- sightings + validation
    recs = load_sightings(region, species, cache_dir)
    rng = np.random.default_rng(42)
    validation = {sid: sightings.validate(recs.get(sid, []), H[sid], grid, rng) for sid in H}
    with open(os.path.join(rdir, "sightings.json"), "w") as f:
        json.dump(recs, f, separators=(",", ":"))
    log.info("habitat check vs GBIF finds (AUC, 0.5 = no skill): " + ", ".join(
        f"{k} {v.get('auc')} (n={v['n']})" for k, v in validation.items()))

    # --- weather
    tz = region.get("timezone", "UTC")
    today = dt.datetime.now(ZoneInfo(tz)).date().isoformat()
    dates, scores, weather_ok = [], {}, False
    lon, lat = grid.cell_lonlat()
    try:
        spacing = region.get("weather_spacing", 0.1)
        lats_, lons_ = weather.lattice(region["bbox"], spacing)
        held = archive.days_held(ARCHIVE_DIR, rid, lats_, lons_)
        past = weather.PAST_DAYS if held >= 60 else archive.SEED_DAYS  # seed a new archive
        if past != weather.PAST_DAYS:
            log.info(f"weather archive has {held} days; seeding with {past} days from Open-Meteo")
        wx = weather.fetch(region["bbox"], spacing, tz, past_days=past)
        try:
            archive.update(ARCHIVE_DIR, rid, wx, today)
        except Exception as exc:
            log.error("weather archive update failed", exc)
        hlon, hlat = downsample(lon, HALF), downsample(lat, HALF)
        helev = downsample(layers["elev"], HALF)
        P, Tmin, Tmax = weather.to_grid(wx, hlon, hlat, helev)
        all_dates = wx["dates"]
        t0 = all_dates.index(today) if today in all_dates else past
        day_idx = list(range(t0, len(all_dates)))
        dates = [all_dates[t] for t in day_idx]
        dry = scoring.dry_run_length(P)
        for d, t in enumerate(day_idx):
            render.save_png(scoring.rain_window(P, t), os.path.join(rdir, "rain", f"{d}.png"),
                            scale=255.0 / 100.0)  # 0..100 mm over 14 days
        for sp in species:
            Ws = scoring.weather_scores(sp, P, Tmin, Tmax, all_dates, day_idx, dry)
            scores[sp["id"]] = []
            for d, W in enumerate(Ws):
                render.save_png(W, os.path.join(rdir, "wx", f"{sp['id']}_{d}.png"))
                s = H[sp["id"]] * upsample(W, grid.shape, HALF)
                scores[sp["id"]].append(s.astype(np.float16))
                render.save_png(s, os.path.join(rdir, "score", f"{sp['id']}_{d}.png"))
        weather_ok = True
    except Exception as exc:
        log.error("weather/scoring failed; publishing habitat only", exc)

    # --- stats + hotspots
    cell_km2 = float(np.mean((grid.res * np.cos(np.radians(lat))) ** 2)) / 1e6
    stats, spots = {}, {}
    for sid, days in scores.items():
        stats[sid] = {
            "max": [round(float(a.max()), 3) for a in days],
            "good_km2": [round(float((a >= GOOD).sum()) * cell_km2, 1) for a in days],
        }
        spots[sid] = [render.hotspots(a.astype(np.float32), grid) for a in days]
    if not scores:
        for sid, h in H.items():
            spots[sid] = [render.hotspots(h, grid)]

    meta = {
        "id": rid, "name": region["name"], "generated": dt.datetime.now(ZoneInfo(tz)).isoformat(timespec="minutes"),
        "grid": grid.to_json(), "half": HALF, "dates": dates, "weather_ok": weather_ok,
        "places": region.get("places", []), "good_threshold": GOOD,
        "stats": stats, "hotspots": spots, "validation": validation,
        "sources": {"tree_species": (None if not all(("ft_" + k) in layers for k in static_layers.FOREST_TYPES)
                                     else "IGN BD Foret" if region.get("country") == "FR" else "ForestPaths EU 10 m"),
                    "ploughing_year": int(layers["pl_year"]) if "pl_year" in layers else None,
                    "rpg": all(("pa_" + k) in layers for k in static_layers.PASTURE_TYPES),
                    "soil_ph": bool(np.isfinite(layers.get("ph", np.array([np.nan]))).any()),
                    "soil_ph_france": bool(layers.get("ph_src_fr", np.array([False])).any()),
                    "pasture_history_year": int(layers["pa_hist_year"]) if "pa_hist_year" in layers else None,
                    "organic_flag": "pa_organic" in layers,
                    "landcover": "lc_grass" in layers},
    }
    with open(os.path.join(rdir, "meta.json"), "w") as f:
        json.dump(meta, f, separators=(",", ":"))
    return grid, dates, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "site"))
    ap.add_argument("--cache", default=os.path.join(ROOT, "cache", "static"))
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    shutil.copytree(os.path.join(ROOT, "site"), args.out)
    os.makedirs(os.path.join(args.out, "data"), exist_ok=True)

    regions = load_yaml("regions.yaml")["regions"]
    species = load_yaml("species.yaml")["species"]
    started = time.time()
    results, index = [], []
    for region in regions:
        try:
            results.append(process_region(region, species, args.out, args.cache))
            index.append({"id": region["id"], "name": region["name"], "bbox": region["bbox"]})
        except Exception as exc:
            log.error(f"region {region['id']} failed", exc)

    public_species = [{k: sp[k] for k in ("id", "name", "latin", "group", "status", "notes", "lookalikes")}
                      for sp in species]
    with open(os.path.join(args.out, "data", "index.json"), "w") as f:
        json.dump({"regions": index, "species": public_species}, f, ensure_ascii=False)

    try:
        names = {sp["id"]: sp["name"] for sp in species}
        lines = alerts.check(alerts.load_spots(), results, names, GOOD, site_url())
        alerts.send(lines, site_url())
    except Exception as exc:
        log.error("alerts failed", exc)

    status = {"finished": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
              "seconds": round(time.time() - started), "log": log.ENTRIES}
    with open(os.path.join(args.out, "data", "status.json"), "w") as f:
        json.dump(status, f, indent=1, ensure_ascii=False)
    if not index:
        raise SystemExit("no region built")


if __name__ == "__main__":
    main()
