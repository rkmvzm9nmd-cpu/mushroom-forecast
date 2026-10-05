"""Push alerts for saved spots via ntfy.sh.

Saved spots come from the SPOTS_JSON repository secret (the site's
"Copy spots for alerts" button produces it) and the channel from NTFY_TOPIC.
Spot names and coordinates are never written to the public log.
"""
from __future__ import annotations

import json
import os

import requests

from . import log


def load_spots():
    raw = os.environ.get("SPOTS_JSON", "").strip()
    if not raw:
        return []
    try:
        spots = json.loads(raw)
        return [s for s in spots if "lat" in s and "lon" in s]
    except Exception as exc:
        log.warn(f"SPOTS_JSON is not valid JSON ({exc.__class__.__name__})")
        return []


def sample(arr, grid, lat, lon, k=1):
    row, col = grid.lonlat_to_cell(lon, lat)
    row, col = int(row), int(col)
    if not (0 <= row < arr.shape[0] and 0 <= col < arr.shape[1]):
        return None
    return float(arr[max(row - k, 0):row + k + 1, max(col - k, 0):col + k + 1].max())


def check(spots, regions_out, species_names, threshold, site_url):
    """regions_out: list of (grid, dates, {species_id: [score arrays per day]})."""
    lines = []
    for spot in spots:
        wanted = spot.get("species") or list(species_names)
        for grid, dates, scores in regions_out:
            best = None
            for sid in wanted:
                for d, arr in enumerate(scores.get(sid, [])):
                    v = sample(arr, grid, spot["lat"], spot["lon"])
                    if v is not None and (best is None or v > best[0]):
                        best = (v, sid, d)
            if best is None:
                continue
            if best[0] >= threshold:
                v, sid, d = best
                when = "today" if d == 0 else dates[d]
                lines.append(f"- {spot.get('name', 'Spot')}: {species_names[sid]} "
                             f"{round(v * 100)}% ({when})")
            break
    return lines


def send(lines, site_url):
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if not topic:
        log.info("alerts: NTFY_TOPIC not set, skipping")
        return
    if not lines:
        log.info("alerts: nothing above threshold today")
        return
    body = "\n".join(lines[:20])
    try:
        r = requests.post(f"https://ntfy.sh/{topic}", data=body.encode("utf-8"), timeout=30, headers={
            "Title": "Mushroom forecast: good conditions at your spots",
            "Tags": "mushroom", "Click": site_url, "Markdown": "yes"})
        log.info(f"alerts: sent {len(lines)} line(s), HTTP {r.status_code}")
    except Exception as exc:
        log.warn(f"alerts: send failed ({exc.__class__.__name__})")
