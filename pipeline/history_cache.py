"""Long-term daily weather cache (Open-Meteo historical archive, ERA5 / ERA5-Land).

Builds a per-point CSV archive so the model can judge the current season against
past ones (drought memory, anomalies, season length) before it has logged its own
history. Two kinds of point per region:

* reference places from config/regions.yaml: daily weather since --start
  (default 1991) plus daily-mean soil moisture for April-November of every year;
* 0.5-degree cells covering the region bounding box (same keys as calibrate.py):
  daily weather since --cell-start (default 2016).

The run is incremental and resumable: files already up to date are skipped, so
if Open-Meteo's daily allowance runs out the job saves what it has and the next
run carries on. Requests are throttled to stay under the free per-minute and
per-hour limits.

    python -m pipeline.history_cache --region massif-central-alps --out history-cache
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
import time

import requests
import yaml

from . import log

API = "https://archive-api.open-meteo.com/v1/archive"
DAILY = ["precipitation_sum", "temperature_2m_max", "temperature_2m_min",
         "temperature_2m_mean", "et0_fao_evapotranspiration"]
SOIL = ["soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm", "soil_moisture_28_to_100cm"]
MINUTE_BUDGET, HOUR_BUDGET = 500, 4200   # free tier: 600/min, 5000/h, 10000/day


class DailyLimit(Exception):
    pass


class Throttle:
    def __init__(self):
        self.calls = []  # (time, weight)

    def wait(self, weight):
        while True:
            now = time.time()
            self.calls = [(t, w) for t, w in self.calls if now - t < 3600]
            minute = sum(w for t, w in self.calls if now - t < 60)
            hour = sum(w for t, w in self.calls)
            if minute + weight <= MINUTE_BUDGET and hour + weight <= HOUR_BUDGET:
                self.calls.append((now, weight))
                return
            log.info(f"throttle: {minute:.0f}/min {hour:.0f}/h used; waiting")
            time.sleep(30)


def weight(start, end, nvars):
    days = (dt.date.fromisoformat(end) - dt.date.fromisoformat(start)).days + 1
    return math.ceil(days / 14) * max(1.0, nvars / 10)


def get(th, params, nvars):
    th.wait(weight(params["start_date"], params["end_date"], nvars))
    for attempt in range(6):
        try:
            r = requests.get(API, params=params, timeout=(20, 300))
        except requests.RequestException as exc:
            log.warn(f"archive {exc.__class__.__name__}; retry")
            time.sleep(30 * (attempt + 1))
            continue
        text = r.text[:200]
        if r.status_code == 200:
            try:
                return r.json()
            except ValueError:
                log.warn(f"archive returned non-JSON ({text!r}); retry")
                time.sleep(60 * (attempt + 1))
                continue
        if r.status_code == 429:
            if "Daily" in text:
                raise DailyLimit(text)
            wait = 3700 if "Hourly" in text else 70
            log.warn(f"archive 429 ({text}); waiting {wait}s")
            time.sleep(wait)
            continue
        log.warn(f"archive HTTP {r.status_code}: {text}")
        if r.status_code == 400:
            return None
        time.sleep(30)
    return None


def cell_key(lat, lon):  # same as calibrate._cell_key
    return f"{round(lat * 2) / 2:.1f}_{round(lon * 2) / 2:.1f}"


def last_date(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = f.read().strip().splitlines()
    return rows[-1].split(",")[0] if len(rows) > 1 else None


def write_rows(path, header, rows, append):
    mode = "a" if append else "w"
    with open(path, mode, newline="") as f:
        w = csv.writer(f)
        if not append:
            w.writerow(header)
        w.writerows(rows)


def fetch_daily(th, path, lat, lon, start, end):
    have = last_date(path)
    if have and have >= end:
        return "up to date"
    s = (dt.date.fromisoformat(have) + dt.timedelta(days=1)).isoformat() if have else start
    rows = []
    # one request per decade keeps each response small
    cur = dt.date.fromisoformat(s)
    stop = dt.date.fromisoformat(end)
    elev = None
    try:
        while cur <= stop:
            e = min(stop, dt.date(cur.year + 9, 12, 31))
            d = get(th, {"latitude": lat, "longitude": lon, "start_date": cur.isoformat(),
                         "end_date": e.isoformat(), "daily": ",".join(DAILY),
                         "timezone": "UTC"}, len(DAILY))
            if d is None:
                break
            elev = d.get("elevation")
            dd = d["daily"]
            for i, t in enumerate(dd["time"]):
                vals = [dd[v][i] for v in DAILY]
                if all(v is None for v in vals):
                    continue
                rows.append([t] + ["" if v is None else v for v in vals])
            cur = e + dt.timedelta(days=1)
    finally:  # keep whatever arrived, even if the allowance ran out mid-way
        write_rows(path, ["date", "P", "Tmax", "Tmin", "Tmean", "ET0"], rows, append=bool(have))
    return f"{len(rows)} days (grid elevation {elev} m)"


def fetch_soil(th, path, lat, lon, y0, end):
    """Daily-mean soil moisture (m3/m3), April-November of each year."""
    have = last_date(path)
    stop = dt.date.fromisoformat(end)
    rows = []
    try:
        _soil_years(th, rows, have, lat, lon, y0, stop)
    finally:
        write_rows(path, ["date", "SM_0_7", "SM_7_28", "SM_28_100"], rows, append=bool(have))
    return f"{len(rows)} days"


def _soil_years(th, rows, have, lat, lon, y0, stop):
    for y in range(y0, stop.year + 1):
        s, e = dt.date(y, 4, 1), min(dt.date(y, 11, 30), stop)
        if have and e.isoformat() <= have:
            continue
        if have and s.isoformat() <= have:
            s = dt.date.fromisoformat(have) + dt.timedelta(days=1)
        if s > e:
            continue
        d = get(th, {"latitude": lat, "longitude": lon, "start_date": s.isoformat(),
                     "end_date": e.isoformat(), "hourly": ",".join(SOIL),
                     "timezone": "UTC"}, len(SOIL))
        if d is None:
            break
        h = d.get("hourly")
        if not h:
            log.warn(f"soil {y}: no hourly block in response ({str(d)[:150]})")
            break
        days = {}
        for i, t in enumerate(h["time"]):
            days.setdefault(t[:10], []).append([h[v][i] for v in SOIL])
        for day, vals in sorted(days.items()):
            out = []
            for k in range(len(SOIL)):
                xs = [v[k] for v in vals if v[k] is not None]
                out.append(round(sum(xs) / len(xs), 4) if len(xs) >= 20 else "")
            if any(o != "" for o in out):
                rows.append([day] + out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--out", default="history-cache")
    ap.add_argument("--start", default="1991-01-01")
    ap.add_argument("--cell-start", default="2016-01-01")
    ap.add_argument("--soil-start", type=int, default=1991)
    ap.add_argument("--parts", default="places,soil,cells")
    a = ap.parse_args()
    cfg = yaml.safe_load(open("config/regions.yaml"))
    reg = next(r for r in cfg["regions"] if r["id"] == a.region)
    end = (dt.date.today() - dt.timedelta(days=2)).isoformat()   # archive lags ~2 days
    out = os.path.join(a.out, reg["id"])
    os.makedirs(out, exist_ok=True)
    parts = a.parts.split(",")
    th = Throttle()
    jobs = []
    for p in reg.get("places", []):
        slug = p["name"].split(" (")[0].replace(" ", "-").replace("/", "-")
        if "places" in parts:
            jobs.append(("daily", os.path.join(out, f"place_{slug}.csv"), p["lat"], p["lon"], a.start))
        if "soil" in parts:
            jobs.append(("soil", os.path.join(out, f"soil_{slug}.csv"), p["lat"], p["lon"], a.soil_start))
    if "cells" in parts:
        w, s, e, n = reg["bbox"]
        lat = math.ceil(s * 2) / 2
        while lat <= n:
            lon = math.ceil(w * 2) / 2
            while lon <= e:
                jobs.append(("daily", os.path.join(out, f"c_{cell_key(lat, lon)}.csv"), lat, lon, a.cell_start))
                lon += 0.5
            lat += 0.5
    log.info(f"{reg['name']}: {len(jobs)} files to check, up to {end}")
    fails = 0
    try:
        for kind, path, lat, lon, start in jobs:
            try:
                if kind == "daily":
                    msg = fetch_daily(th, path, lat, lon, start, end)
                else:
                    msg = fetch_soil(th, path, lat, lon, start, end)
                log.info(f"{os.path.basename(path)}: {msg}")
                fails = 0
            except DailyLimit as exc:
                log.warn(f"daily allowance used up ({exc}); stopping, next run resumes")
                break
            except Exception as exc:  # keep going; the log says what broke
                log.error(f"{os.path.basename(path)} failed", exc)
                fails += 1
                if fails >= 3:
                    log.warn("three failures in a row; stopping, next run resumes")
                    break
    finally:
        # the run log travels with the data, so failures can be read without the Actions UI
        tag = a.parts.replace(",", "+")
        with open(os.path.join(out, f"_log_{tag}.txt"), "a") as f:
            f.write(f"=== run {dt.datetime.utcnow():%Y-%m-%d %H:%M} UTC, parts {a.parts}\n")
            for e in log.ENTRIES:
                f.write(f"[{e['t']}] {e['level'].upper():5s} {e['msg']}\n")


if __name__ == "__main__":
    main()
