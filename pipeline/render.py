"""Write score grids as 8-bit greyscale PNGs (the site colours them) and find hotspots."""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter, uniform_filter


def save_png(arr01, path, scale=255.0):
    a = np.nan_to_num(np.asarray(arr01, dtype=np.float32))
    img = np.clip(np.round(a * scale), 0, 255).astype(np.uint8)
    Image.fromarray(img, mode="L").save(path, optimize=True)


def hotspots(score, grid, n=8, min_score=0.25, radius_px=25):
    """Top-n local maxima of the smoothed score, at least ~radius apart."""
    sm = uniform_filter(score.astype(np.float32), size=5)
    peaks = (sm == maximum_filter(sm, size=2 * radius_px + 1)) & (sm >= min_score)
    rows, cols = np.nonzero(peaks)
    if rows.size == 0:
        return []
    vals = sm[rows, cols]
    order = np.argsort(-vals)[: n * 3]
    chosen = []
    for i in order:
        r, c = rows[i], cols[i]
        if all((r - r2) ** 2 + (c - c2) ** 2 > radius_px ** 2 for r2, c2, _ in chosen):
            chosen.append((r, c, vals[i]))
        if len(chosen) >= n:
            break
    lon, lat = grid.cell_lonlat()
    return [{"lat": round(float(lat[r, c]), 5), "lon": round(float(lon[r, c]), 5),
             "score": round(float(v), 3)} for r, c, v in chosen]
