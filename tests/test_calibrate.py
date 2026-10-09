"""Offline test of the calibration pipeline with synthetic data where the 'truth' is known:
finds sit where ft_beech is high, and fruit ~8-14 days after rain at 8-12 C."""
import datetime as dt
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests import test_local as TL  # noqa: E402  (reuses fake layers; runs its own smoke test first)
from pipeline import calibrate as C, run as R, sightings, static_layers  # noqa: E402
from pipeline.grid import Grid  # noqa: E402


def fake_jjas(points, year):
    # 2026 is the driest year everywhere; other years vary
    return [{"rain": 300.0, "balance": -400.0 if year == 2026 else -150.0 + 7 * (year % 11)} for _ in points]


C._jjas = fake_jjas
C.time.sleep = lambda s: None

rng = np.random.default_rng(3)
static_layers.GROUPS = TL.fake_groups()
orig = R.load_yaml
def small(name):
    d = orig(name)
    if name == "regions.yaml":
        d["regions"] = [d["regions"][0]]
        d["regions"][0]["bbox"] = [3.6, 44.65, 3.9, 44.8]
    return d
R.load_yaml = small
region = R.load_yaml("regions.yaml")["regions"][0]
grid = Grid.from_bbox(region["bbox"], 10)
lon, lat = grid.cell_lonlat()

# synthetic history: one cell, rain pulses; finds placed 8-14 days after big rain when T ~ 10
days = (dt.date(2026, 9, 30) - dt.date(2020, 1, 1)).days
P = np.where(rng.random(days) < 0.12, rng.uniform(15, 40, days), 0).astype(np.float32)
T = (10 + 8 * np.sin(np.arange(days) / 365 * 2 * np.pi - 1.8) + rng.normal(0, 2, days)).astype(np.float32)
hist = {C._cell_key(44.7, 3.75): {"start": "2020-01-01", "P": P, "T": T, "Tmin": T - 5}}
good = [i for i in range(60, days) if P[i - 14:i - 7].sum() > 25 and 8 < T[i - 6:i + 1].mean() < 13]
print("candidate find days", len(good)); sel = rng.choice(good, min(300, len(good)), replace=False)
# habitat truth: beech fraction from fake layers is uniform; use grass mask (alternating columns)
grass = (np.sin(np.arange(grid.width) / 25) > 0)
cols = np.where(~grass)[0]
recs = []
for i, d in enumerate(sel):
    c = int(rng.choice(cols)); r = int(rng.integers(0, grid.height))
    recs.append({"lat": float(lat[r, c]), "lon": float(lon[r, c]),
                 "date": (dt.date(2020, 1, 1) + dt.timedelta(days=int(d))).isoformat(), "month": 10, "id": i})
bg = [{"lat": float(lat[r, c]), "lon": float(lon[r, c]), "date": "2022-10-01", "month": 10}
      for r, c in zip(rng.integers(0, grid.height, 800), rng.integers(0, grid.width, 800))]
sightings.fetch = lambda species, bbox: {sp["id"]: (recs if sp["id"] == "boletus" else recs[:20]) for sp in species}
C.gbif_background = lambda bbox, per_year=300: bg
C.month_counts = lambda bbox, keys: np.array([1, 1, 1, 1, 2, 5, 8, 15, 40, 60, 20, 3], float) * (2 if keys == [C.FUNGI_KEY] else 1)
C.species_keys = lambda sp: [1]
C.load_history = lambda keys, d, end: hist
C._cell_key = lambda la, lo: "44.7_3.8" if True else None
hist["44.7_3.8"] = hist.pop(list(hist)[0])

with tempfile.TemporaryDirectory() as tmp:
    sys.argv = ["cal", "--cache", os.path.join(tmp, "cache"), "--out", os.path.join(tmp, "cal"), "--history", tmp]
    C.main()
    cal = json.load(open(os.path.join(tmp, "cal", "calibration.json")))
    b = cal["habitat"]["massif-central-alps"]["boletus"]
    print("habitat boletus:", {k: b.get(k) for k in ("n", "auc_rule", "auc_fit_cv", "use")}, b.get("top_factors", [])[:3])
    t = cal["timing"]["FR"]["boletus"]
    print("timing boletus:", {k: t.get(k) for k in ("n_train", "n_test", "auc_rule_test", "auc_fit_test", "params", "use")})
    print("season boletus:", cal["season"]["FR"]["boletus"].get("weights"))
    print(open(os.path.join(tmp, "cal", "REPORT.md")).read()[:600])
    # apply in a run
    import shutil
    os.makedirs(os.path.join(R.ROOT, "calibration"), exist_ok=True)
print("OK")


# drought-year check
clim = C.drought_years(R.load_yaml("regions.yaml")["regions"], os.path.join(tempfile.mkdtemp(), "climate.json"),
                       today=dt.date(2026, 10, 9))
r = clim["regions"]["massif-central-alps"]
assert r["year"] == 2026 and "Chastanier" in r["drought_places"], r
assert r["places"]["Chastanier"]["rank_driest"] == 1 and r["places"]["Chastanier"]["of_years"] == 36
print("drought OK")
