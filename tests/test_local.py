"""Offline smoke test: fake static layers + fake weather, run the whole pipeline twice
(first run seeds the weather archive, second run uses it)."""
import datetime as dt
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import bucket, run, static_layers, weather, sightings, soil_moisture  # noqa: E402

calls = []


def fake_groups():
    def dem(r, g, w):
        h, wd = g.shape
        yy, xx = np.mgrid[0:h, 0:wd]
        return {"elev": (800 + 300 * np.sin(xx / 40) * np.cos(yy / 30)).astype(np.float32)}

    def lc(r, g, w):
        h, wd = g.shape
        grass = (np.sin(np.mgrid[0:h, 0:wd][1] / 25) > 0).astype(np.float32)
        z = np.zeros_like(grass, np.float16)
        return {"lc_tree": (1 - grass).astype(np.float16), "lc_grass": grass.astype(np.float16),
                "lc_shrub": z, "lc_crop": z, "lc_built": z, "lc_bare": z, "lc_water": z}

    def ph(r, g, w):
        return {"ph": (5 + np.random.default_rng(0).random(g.shape)).astype(np.float32)}

    def bdf(r, g, w):
        lcl = lc(r, g, w)
        return {"ft_" + k: (lcl["lc_tree"] / 8).astype(np.float16) for k in static_layers.FOREST_TYPES}

    def rpg(r, g, w):
        grass = lc(r, g, w)["lc_grass"].astype(np.float32)
        return {"pa_permanent": (grass * 0.8).astype(np.float16), "pa_rough": np.zeros_like(grass, np.float16),
                "pa_temporary": (grass * 0.3).astype(np.float16), "pa_tilled": (1 - grass).astype(np.float16),
                "pa_organic": (grass * 0.1).astype(np.float16)}

    def hist(r, g, w):
        grass = lc(r, g, w)["lc_grass"].astype(np.float32)
        return {"pa_hist_perm": (grass * 0.4).astype(np.float16), "pa_hist_year": np.array(2015)}

    def ndmi(r, g, w):
        rng = np.random.default_rng(1)
        return {"ndmi": rng.normal(0.2, 0.1, g.shape).astype(np.float16),
                "ndmi_age": np.full(g.shape, 5, np.uint8), "ndmi_built": np.array("2026-10-08")}

    def bare(r, g, w):
        grass = lc(r, g, w)["lc_grass"].astype(np.float32)
        h, wd = g.shape
        tilled = (np.mgrid[0:h, 0:wd][0] < h // 3) * grass       # top third of the grass "was tilled"
        crop = lc(r, g, w)["lc_tree"].astype(np.float32)          # pretend: tree pixels are declared arable
        return {"bare_frac": (0.5 * tilled + 0.6 * crop).astype(np.float16), "bare_views": np.full(g.shape, 6, np.uint8),
                "bare_built": np.array("2026-10-08")}

    return {"bare_soil": (1, bare, 30), "ndmi": (1, ndmi, 5), "dem": (1, dem), "lc": (1, lc), "ph": (2, ph), "bdforet": (1, bdf), "rpg": (3, rpg), "rpg_hist": (1, hist)}


def fake_fetch(bbox, spacing, tz, batch=25, past_days=31):
    calls.append(past_days)
    lats, lons = weather.lattice(bbox, spacing)
    today = dt.date.today()
    dates = [(today + dt.timedelta(days=d)).isoformat() for d in range(-past_days, 9)]
    T = len(dates)
    P = np.zeros((T, len(lats), len(lons)), np.float32)
    P[T - 15:T - 12] = 15
    return {"dates": dates, "lats": lats, "lons": lons, "elev": np.full((len(lats), len(lons)), 900, np.float32),
            "P": P, "Tmax": np.full_like(P, 15.0), "Tmin": np.full_like(P, 6.0), "ET0": np.full_like(P, 2.0)}


def fake_gbif(species, bbox):
    return {sp["id"]: [{"lat": 44.75, "lon": 3.75, "date": "2025-10-10", "month": 10, "id": 1}] * 6 for sp in species}


def fake_cells(bbox, store_dir, today):
    lats, lons = bucket.cell_axes(bbox)
    n = 300
    # dry summer: no rain, 4 mm/day evaporation -> deficit near the 150 mm cap
    cells = {bucket._key(la, lo): {"start": (dt.date.fromisoformat(today) - dt.timedelta(days=n)).isoformat(),
                                   "P": np.zeros(n, np.float32), "ET0": np.full(n, 4.0, np.float32)}
             for la in lats for lo in lons}
    return lats, lons, cells


bucket.load_cells = fake_cells
static_layers.GROUPS = fake_groups()
soil_moisture.latest = lambda bbox, lon, lat, today: (
    np.full(lon.shape, 30.0, np.float32), (dt.date.fromisoformat(today) - dt.timedelta(days=2)).isoformat())
weather.fetch = fake_fetch
sightings.fetch = fake_gbif
os.environ["SPOTS_JSON"] = json.dumps([{"name": "test", "lat": 44.75, "lon": 3.75}])

with tempfile.TemporaryDirectory() as tmp:
    run.ARCHIVE_DIR = os.path.join(tmp, "archive")
    orig = run.load_yaml

    def small(name):
        d = orig(name)
        if name == "regions.yaml":
            d["regions"][0]["bbox"] = [3.6, 44.65, 3.9, 44.8]
            d["regions"][1]["bbox"] = [24.7, 45.7, 24.9, 45.85]
            d["regions"][2]["bbox"] = [-3.4, 55.5, -3.2, 55.65]
            d["regions"][3]["bbox"] = [-3.9, 57.1, -3.7, 57.25]
        return d
    run.load_yaml = small
    for _ in range(2):
        sys.argv = ["run", "--out", os.path.join(tmp, "site"), "--cache", os.path.join(tmp, "cache")]
        run.main()
    meta = json.load(open(os.path.join(tmp, "site/data/massif-central-alps/meta.json")))
    ro = json.load(open(os.path.join(tmp, "site/data/brasov-fagaras/meta.json")))
    print("satellite", ro["satellite_wetness"], {k: (v.get("auc"), v.get("auc_with_satellite"), v.get("satellite_used")) for k, v in ro["validation"].items()})
    print("romania sources", ro["sources"], "dates", ro["dates"][:2], "soil", ro["soil_date"])
    assert os.path.exists(os.path.join(tmp, "site/data/brasov-fagaras/wx/soil.png"))
    import pipeline.habitat as H
    lay = {"lc_grass": np.ones((2, 2), np.float16), "lc_shrub": np.zeros((2, 2), np.float16),
           "pl_recent": np.array([[1, 0], [0, 0]], np.float16), "pl_mid": np.array([[0, 1], [0, 0]], np.float16),
           "pl_changed": np.zeros((2, 2), np.float16),
           "mw_two": np.zeros((2, 2), np.float16), "mw_many": np.array([[0, 0], [1, 0]], np.float16)}
    sp = [s_ for s_ in run.load_yaml("species.yaml")["species"] if s_["id"] == "psilocybe"][0]
    print("plough effect (recent, mid / mown 3+, none):", H.host_score(sp, lay, (2, 2)).tolist())
    print("past_days requested per run:", calls)
    print("sources", meta["sources"])
    print("max scores", {k: v["max"][:4] for k, v in meta["stats"].items()})
    assert meta["weather_ok"] and calls == [92, 92, 92, 92, 31, 31, 31, 31] and meta["sources"]["rpg"]
print("OK")


# --- root-zone bucket unit checks (2026 drought analysis rules)
D = bucket.run(np.array([[0.0], [40.0]], np.float32), np.array([[3.0], [1.0]], np.float32), np.array([120.0], np.float32))
assert abs(D[0, 0] - min(150, 120 + 3 * (1 - 120 / 150))) < 1e-4
before = bucket.before_each_day(D, np.array([120.0], np.float32))
pe = bucket.effective_rain(np.array([[0.0], [40.0]], np.float32), before, 50)
assert pe[1, 0] == 0.0, pe          # 40 mm on a 120 mm deficit only refills the soil
pe2 = bucket.effective_rain(np.array([[40.0]], np.float32), np.array([[60.0]], np.float32), 50)
assert abs(pe2[0, 0] - 30.0) < 1e-4  # 10 mm refills to D0, 30 mm counts
from pipeline import scoring  # noqa: E402
sp = [x for x in run.load_yaml("species.yaml")["species"] if x["id"] == "psilocybe"][0]
T, Hh, Ww = 40, 2, 2
Pw = np.zeros((T, Hh, Ww), np.float32); Pw[20:23] = 20.0
Tmin = np.full_like(Pw, 6.0); Tmax = np.full_like(Pw, 12.0)
dates = [(dt.date(2026, 10, 1) + dt.timedelta(days=i)).isoformat() for i in range(T)]
wet = scoring.weather_scores(sp, Pw, Tmin, Tmax, dates, [36], deficit=np.zeros_like(Pw))[0]
dry = scoring.weather_scores(sp, Pw, Tmin, Tmax, dates, [36], deficit=np.full_like(Pw, 120.0))[0]
assert wet.max() > 0.1 and dry.max() == 0.0, (wet.max(), dry.max())
print("bucket OK")
