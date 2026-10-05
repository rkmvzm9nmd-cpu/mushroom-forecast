"""Public sighting records from GBIF (includes iNaturalist research-grade and
national records), used to (a) show past finds on the map and (b) check how well
the habitat model separates real finds from random ground."""
from __future__ import annotations

import time

import numpy as np
import requests

from . import log

API = "https://api.gbif.org/v1"
MAX_PER_NAME = 1500


def _taxon_key(name):
    r = requests.get(f"{API}/species/match", params={"name": name, "kingdom": "Fungi"}, timeout=60)
    r.raise_for_status()
    d = r.json()
    return d.get("usageKey") if d.get("matchType") != "NONE" else None


def fetch(species_list, bbox):
    w, s, e, n = bbox
    out = {}
    for sp in species_list:
        recs = []
        for name in sp.get("gbif", []):
            try:
                key = _taxon_key(name)
                if not key:
                    log.warn(f"GBIF: no match for {name}")
                    continue
                offset = 0
                while offset < MAX_PER_NAME:
                    r = requests.get(f"{API}/occurrence/search", timeout=90, params={
                        "taxonKey": key, "hasCoordinate": "true", "hasGeospatialIssue": "false",
                        "occurrenceStatus": "PRESENT",
                        "decimalLatitude": f"{s},{n}", "decimalLongitude": f"{w},{e}",
                        "limit": 300, "offset": offset})
                    r.raise_for_status()
                    d = r.json()
                    for o in d.get("results", []):
                        unc = o.get("coordinateUncertaintyInMeters")
                        if unc is not None and unc > 1000:
                            continue
                        recs.append({
                            "lat": round(o["decimalLatitude"], 5), "lon": round(o["decimalLongitude"], 5),
                            "date": (o.get("eventDate") or "")[:10], "month": o.get("month"),
                            "id": o.get("gbifID") or o.get("key"),
                        })
                    if d.get("endOfRecords", True):
                        break
                    offset += 300
                    time.sleep(0.3)
            except Exception as exc:
                log.warn(f"GBIF {name}: {exc.__class__.__name__}: {str(exc)[:150]}")
        # de-duplicate identical coordinates+date
        seen, uniq = set(), []
        for r_ in recs:
            k = (r_["lat"], r_["lon"], r_["date"])
            if k not in seen:
                seen.add(k)
                uniq.append(r_)
        out[sp["id"]] = uniq
        log.info(f"GBIF {sp['id']}: {len(uniq)} usable records in region")
    return out


def auc(scores_pos, scores_bg):
    """Probability a random real find scores higher than a random background cell."""
    if len(scores_pos) == 0 or len(scores_bg) == 0:
        return None
    from scipy.stats import rankdata
    ranks = rankdata(np.concatenate([scores_pos, scores_bg]))
    n1, n2 = len(scores_pos), len(scores_bg)
    u = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    return float(u / (n1 * n2))


def validate(recs, hab, grid, rng):
    """AUC of the habitat layer: real finds vs random land cells."""
    pts = []
    for r_ in recs:
        row, col = grid.lonlat_to_cell(r_["lon"], r_["lat"])
        row, col = int(row), int(col)
        if 0 <= row < hab.shape[0] and 0 <= col < hab.shape[1]:
            pts.append(hab[max(row - 1, 0):row + 2, max(col - 1, 0):col + 2].max())
    if len(pts) < 5:
        return {"n": len(pts), "auc": None}
    bg = hab[rng.integers(0, hab.shape[0], 5000), rng.integers(0, hab.shape[1], 5000)]
    return {"n": len(pts), "auc": round(auc(np.array(pts), bg), 3),
            "mean_at_finds": round(float(np.mean(pts)), 3), "mean_background": round(float(bg.mean()), 3)}
