"""Offline smoke test: fake static layers + fake weather, run the whole pipeline."""
import datetime as dt
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import run, static_layers, weather, sightings  # noqa: E402
from pipeline.grid import Grid  # noqa: E402


def fake_static(region, grid, path, workdir):
    rng = np.random.default_rng(0)
    h, w = grid.shape
    yy, xx = np.mgrid[0:h, 0:w]
    layers = {"elev": (800 + 300 * np.sin(xx / 40) * np.cos(yy / 30)).astype(np.float32)}
    grass = (np.sin(xx / 25) > 0).astype(np.float32)
    layers.update(lc_tree=(1 - grass).astype(np.float16), lc_grass=grass.astype(np.float16),
                  lc_shrub=np.zeros_like(grass, np.float16), lc_crop=np.zeros_like(grass, np.float16),
                  lc_built=np.zeros_like(grass, np.float16), lc_bare=np.zeros_like(grass, np.float16),
                  lc_water=np.zeros_like(grass, np.float16))
    layers["ph"] = (5 + rng.random(grid.shape)).astype(np.float32)
    for k in static_layers.FOREST_TYPES:
        layers["ft_" + k] = ((1 - grass) / len(static_layers.FOREST_TYPES)).astype(np.float16)
    np.savez_compressed(path, **layers)
    return layers


def fake_fetch(bbox, spacing, tz, batch=50):
    lats, lons = weather.lattice(bbox, spacing)
    today = dt.date.today()
    dates = [(today + dt.timedelta(days=d)).isoformat() for d in range(-31, 9)]
    T = len(dates)
    P = np.zeros((T, len(lats), len(lons)), np.float32)
    P[25:28] = 15  # a soaking ~5 days ago
    Tmax = np.full_like(P, 15.0)
    Tmin = np.full_like(P, 6.0)
    return {"dates": dates, "lats": lats, "lons": lons,
            "elev": np.full((len(lats), len(lons)), 900, np.float32), "P": P, "Tmax": Tmax, "Tmin": Tmin}


def fake_gbif(species, bbox):
    return {sp["id"]: [{"lat": 44.75, "lon": 3.75, "date": "2025-10-10", "month": 10, "id": 1}] * 6 for sp in species}


static_layers.build = fake_static
weather.fetch = fake_fetch
sightings.fetch = fake_gbif
os.environ["SPOTS_JSON"] = json.dumps([{"name": "test", "lat": 44.75, "lon": 3.75}])

with tempfile.TemporaryDirectory() as tmp:
    orig = run.load_yaml
    def small(name):
        d = orig(name)
        if name == "regions.yaml":
            d["regions"][0]["bbox"] = [3.6, 44.65, 3.9, 44.8]
        return d
    run.load_yaml = small
    sys.argv = ["run", "--out", os.path.join(tmp, "site"), "--cache", os.path.join(tmp, "cache")]
    run.main()
    meta = json.load(open(os.path.join(tmp, "site/data/massif-central-alps/meta.json")))
    print("dates", meta["dates"][:3], "weather_ok", meta["weather_ok"])
    print("max scores", {k: v["max"][:3] for k, v in meta["stats"].items()})
    print("hotspots psilocybe d0", meta["hotspots"]["psilocybe"][0][:2])
    print("files", sorted(os.listdir(os.path.join(tmp, "site/data/massif-central-alps/score")))[:5])
    assert meta["weather_ok"]
print("OK")
