"""Web-Mercator pixel grid shared by every layer of a region.

The grid is aligned to the standard slippy-map pixel pyramid at `zoom`, so a PNG
written on it drops straight onto a Leaflet map as an image overlay with no
reprojection error.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

HALF = 20037508.342789244  # half the Web-Mercator world width, metres


def lonlat_to_px(lon, lat, zoom):
    n = 256 * 2 ** zoom
    lon = np.asarray(lon, dtype=float)
    lat = np.clip(np.asarray(lat, dtype=float), -85.05112878, 85.05112878)
    x = (lon + 180.0) / 360.0 * n
    phi = np.radians(lat)
    y = (1.0 - np.log(np.tan(phi) + 1.0 / np.cos(phi)) / math.pi) / 2.0 * n
    return x, y


def px_to_lonlat(x, y, zoom):
    n = 256 * 2 ** zoom
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lon = x / n * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lon, lat


@dataclass
class Grid:
    zoom: int
    x0: int       # global pixel column of the left edge
    y0: int       # global pixel row of the top edge
    width: int
    height: int
    factor: int = 1   # sub-sampling factor (grid is `factor` times finer than the zoom pixels)

    @classmethod
    def from_bbox(cls, bbox, zoom):
        west, south, east, north = bbox
        xa, ya = lonlat_to_px(west, north, zoom)
        xb, yb = lonlat_to_px(east, south, zoom)
        x0, y0 = int(math.floor(xa)), int(math.floor(ya))
        return cls(zoom, x0, y0, int(math.ceil(xb)) - x0, int(math.ceil(yb)) - y0)

    @property
    def shape(self):
        return (self.height, self.width)

    @property
    def res(self):
        """Pixel size in Web-Mercator metres."""
        return 2 * HALF / (256 * 2 ** self.zoom) / self.factor

    @property
    def origin(self):
        """Top-left corner in EPSG:3857 metres."""
        base = 2 * HALF / (256 * 2 ** self.zoom)
        return (self.x0 * base - HALF, HALF - self.y0 * base)

    @property
    def transform(self):
        from rasterio.transform import Affine  # imported lazily (only needed in CI)
        ox, oy = self.origin
        return Affine(self.res, 0.0, ox, 0.0, -self.res, oy)

    @property
    def bounds_3857(self):
        ox, oy = self.origin
        return (ox, oy - self.height * self.res, ox + self.width * self.res, oy)

    @property
    def latlon_bounds(self):
        """[[south, west], [north, east]] for Leaflet."""
        f = self.factor
        w, n = px_to_lonlat(self.x0, self.y0, self.zoom)
        e, s = px_to_lonlat(self.x0 + self.width / f, self.y0 + self.height / f, self.zoom)
        return [[float(s), float(w)], [float(n), float(e)]]

    def finer(self, k):
        return Grid(self.zoom, self.x0, self.y0, self.width * k, self.height * k, self.factor * k)

    def cell_lonlat(self):
        """Lon/lat of every cell centre, as two (H, W) arrays."""
        f = self.factor
        cols = self.x0 + (np.arange(self.width) + 0.5) / f
        rows = self.y0 + (np.arange(self.height) + 0.5) / f
        lon, _ = px_to_lonlat(cols, np.full_like(cols, self.y0), self.zoom)
        _, lat = px_to_lonlat(np.full_like(rows, self.x0), rows, self.zoom)
        return np.broadcast_to(lon[None, :], self.shape), np.broadcast_to(lat[:, None], self.shape)

    def lonlat_to_cell(self, lon, lat):
        """Fractional (row, col) of a lon/lat point on this grid."""
        x, y = lonlat_to_px(lon, lat, self.zoom)
        return (y - self.y0) * self.factor, (x - self.x0) * self.factor

    def to_json(self):
        return {
            "zoom": self.zoom, "x0": self.x0, "y0": self.y0,
            "width": self.width, "height": self.height,
            "bounds": self.latlon_bounds,
        }


def block_mean(a, k):
    """Average k x k blocks (array dims must be multiples of k)."""
    h, w = a.shape[0] // k, a.shape[1] // k
    return a[: h * k, : w * k].reshape(h, k, w, k).mean(axis=(1, 3))


def downsample(a, k):
    """Mean-downsample by k, padding edges so any shape works."""
    h, w = a.shape
    H, W = -(-h // k) * k, -(-w // k) * k
    pad = np.pad(a.astype(np.float32), ((0, H - h), (0, W - w)), mode="edge")
    return block_mean(pad, k)


def upsample(a, shape, k):
    """Bilinear upsample an array made by downsample(x, k) back to `shape`."""
    from scipy.ndimage import map_coordinates
    H, W = shape
    yy = (np.arange(H) + 0.5) / k - 0.5
    xx = (np.arange(W) + 0.5) / k - 0.5
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    return map_coordinates(a.astype(np.float32), [Y, X], order=1, mode="nearest")
