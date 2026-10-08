"""Per-cell habitat features shared by calibration (fitting) and the daily run (prediction).

The learned habitat model is a regularised logistic regression on these features,
including the hand-set rule score itself, so with little data it stays close to the
rules and with lots of data it can re-weight tree types, pH, altitude, grass types etc.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

from .static_layers import FOREST_TYPES

GRASS_KEYS = ["pa_permanent", "pa_rough", "pa_temporary", "pa_tilled", "pl_recent", "pl_mid", "pl_changed"]
LC_KEYS = ["lc_grass", "lc_shrub", "lc_crop", "lc_built", "lc_water", "lc_bare"]


def _get(layers, k, shape):
    a = layers.get(k)
    return np.zeros(shape, np.float32) if a is None else np.nan_to_num(a.astype(np.float32))


def shared_features(layers, terr, sat_wet, shape):
    """Dict name -> (H, W) float32, everything except the species' rule score."""
    slope, damp = terr
    f = {}
    for k in FOREST_TYPES:
        f["ft_" + k] = _get(layers, "ft_" + k, shape)
    tree = _get(layers, "lc_tree", shape)
    f["unmapped_tree"] = np.clip(tree - sum(f["ft_" + k] for k in FOREST_TYPES), 0, 1)
    for k in LC_KEYS + GRASS_KEYS:
        f[k] = _get(layers, k, shape)
    ph = layers.get("ph")
    if ph is None:
        ph = np.full(shape, 5.5, np.float32)
    ph = np.where(np.isfinite(ph), ph, np.nanmedian(ph) if np.isfinite(ph).any() else 5.5).astype(np.float32)
    f["ph"] = ph
    f["ph_sq"] = (ph - 5.5) ** 2
    e = layers["elev"]
    e = np.where(np.isfinite(e), e, np.nanmedian(e) if np.isfinite(e).any() else 500) / 1000.0
    f["elev_km"] = e.astype(np.float32)
    f["elev_sq"] = ((e - 0.8) ** 2).astype(np.float32)
    f["slope"] = (slope / 45.0).astype(np.float32)
    f["damp"] = damp.astype(np.float32)
    f["satwet"] = (np.nan_to_num(sat_wet, nan=0.5) if sat_wet is not None
                   else np.full(shape, 0.5)).astype(np.float32)
    return f


def smooth(feats, size=3):
    """3x3 mean, to absorb ~100-300 m location error of public records."""
    return {k: uniform_filter(v, size=size) for k, v in feats.items()}


def predict(model, rule, shared):
    """Apply a fitted model (dict from calibration) to full grids -> 0..1 habitat."""
    z = np.full(rule.shape, model["intercept"], np.float32)
    for name, c, m, s in zip(model["features"], model["coef"], model["mean"], model["std"]):
        x = rule if name == "rule" else shared.get(name)
        if x is None:
            x = np.full(rule.shape, m, np.float32)
        z += np.float32(c / s) * (x - np.float32(m))
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    return np.clip(p / max(model["scale"], 1e-6), 0, 1).astype(np.float32)
