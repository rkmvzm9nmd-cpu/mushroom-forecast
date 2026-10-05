"""Static (rarely changing) habitat inputs, resampled onto a region's grid.

Sources (all open, no account needed):
  * Elevation   - Copernicus DEM GLO-30, public COGs on AWS
  * Land cover  - ESA WorldCover 2021 (10 m), public COGs on AWS
  * Soil pH     - ISRIC SoilGrids 2.0 (250 m), WCS
  * Tree species (France only) - IGN BD Foret V2 via the Geoplateforme WFS

Each source fails soft: if one is unreachable the run continues with a neutral
fallback and a warning in status.json.
"""
from __future__ import annotations

import math
import os
import time
import unicodedata

import numpy as np
import requests

from . import log
from .grid import Grid, block_mean

UA = {"User-Agent": "mushroom-forecast (github.com/rkmvzm9nmd-cpu/mushroom-forecast)"}

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")

FOREST_TYPES = ["beech", "oak", "chestnut", "pine", "sprucefir", "mixedbroad", "mixed", "otherbroad"]
LC_CLASSES = {"tree": [10], "shrub": [20], "grass": [30], "crop": [40], "built": [50],
              "bare": [60, 70], "water": [80, 90, 95, 100]}
SUB = 4  # sub-sampling factor for fractional land cover / forest type


# --------------------------------------------------------------------------- helpers
def _reproject_tiles(urls, grid: Grid, resampling, dtype, nodata):
    import rasterio
    from rasterio.warp import reproject
    from rasterio.crs import CRS

    dst = np.full(grid.shape, nodata, dtype=dtype)
    used = 0
    for url in urls:
        try:
            with rasterio.open(url) as src:
                reproject(
                    source=rasterio.band(src, 1), destination=dst,
                    src_nodata=src.nodata, dst_nodata=nodata,
                    dst_transform=grid.transform, dst_crs=CRS.from_epsg(3857),
                    resampling=resampling, init_dest_nodata=False,
                    num_threads=4, warp_mem_limit=1024,
                )
                used += 1
        except Exception as exc:  # missing tile (e.g. sea) or network error
            log.warn(f"tile skipped {url.rsplit('/', 1)[-1]}: {exc.__class__.__name__}: {str(exc)[:160]}")
    return dst, used


def _tile_range(lo, hi, step):
    return range(int(math.floor(lo / step) * step), int(math.ceil(hi / step) * step), step)


def _ns(v):
    return f"N{v:02d}" if v >= 0 else f"S{-v:02d}"


def _ew(v, w=3):
    return f"E{v:0{w}d}" if v >= 0 else f"W{-v:0{w}d}"


# --------------------------------------------------------------------------- elevation
def elevation(bbox, grid: Grid):
    from rasterio.enums import Resampling
    w, s, e, n = bbox
    urls = []
    for lat in _tile_range(s, n, 1):
        for lon in _tile_range(w, e, 1):
            name = f"Copernicus_DSM_COG_10_{_ns(lat)}_00_{_ew(lon)}_00_DEM"
            urls.append(f"https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif")
    dem, used = _reproject_tiles(urls, grid, Resampling.average, np.float32, -9999.0)
    dem[dem <= -9000] = np.nan
    log.info(f"elevation: {used}/{len(urls)} DEM tiles, range {np.nanmin(dem):.0f}-{np.nanmax(dem):.0f} m")
    return dem


# --------------------------------------------------------------------------- land cover
def landcover(bbox, grid: Grid):
    from rasterio.enums import Resampling
    w, s, e, n = bbox
    urls = []
    for lat in _tile_range(s, n, 3):
        for lon in _tile_range(w, e, 3):
            tile = f"{_ns(lat)}{_ew(lon)}"
            urls.append("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
                        f"ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")
    fine = grid.finer(SUB)
    lc, used = _reproject_tiles(urls, fine, Resampling.nearest, np.uint8, 0)
    out = {}
    for name, codes in LC_CLASSES.items():
        out[name] = block_mean(np.isin(lc, codes).astype(np.float32), SUB).astype(np.float16)
    log.info(f"landcover: {used}/{len(urls)} WorldCover tiles; "
             + ", ".join(f"{k} {float(v.mean()) * 100:.0f}%" for k, v in out.items()))
    return out


# --------------------------------------------------------------------------- soil pH
def soil_ph(bbox, grid: Grid, workdir):
    from rasterio.enums import Resampling
    w, s, e, n = bbox
    pad = 0.05
    base = "https://maps.isric.org/mapserv?map=/map/phh2o.map&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
    crs = "http://www.opengis.net/def/crs/EPSG/0/4326"
    variants = [
        f"{base}&COVERAGEID=phh2o_0-5cm_mean&FORMAT=image/tiff"
        f"&SUBSET=long({w - pad},{e + pad})&SUBSET=lat({s - pad},{n + pad})"
        f"&SUBSETTINGCRS={crs}&OUTPUTCRS={crs}",
        f"{base}&COVERAGEID=phh2o_0-5cm_mean&FORMAT=GEOTIFF_INT16"
        f"&SUBSET=X({w - pad},{e + pad})&SUBSET=Y({s - pad},{n + pad})"
        f"&SUBSETTINGCRS={crs}&OUTPUTCRS={crs}",
    ]
    path = os.path.join(workdir, "phh2o.tif")
    for url in variants:
        try:
            r = requests.get(url, headers=UA, timeout=180)
            if r.status_code == 200 and r.content[:2] in (b"II", b"MM"):
                with open(path, "wb") as f:
                    f.write(r.content)
                ph, used = _reproject_tiles([path], grid, Resampling.bilinear, np.float32, -1.0)
                ph[(ph <= 0) | (ph > 140)] = np.nan
                ph = ph / 10.0  # SoilGrids stores pH x 10
                log.info(f"soil pH: SoilGrids ok, range {np.nanmin(ph):.1f}-{np.nanmax(ph):.1f}")
                return ph
            log.warn(f"soil pH: HTTP {r.status_code}, {r.content[:120]!r}")
        except Exception as exc:
            log.warn(f"soil pH request failed: {exc.__class__.__name__}: {str(exc)[:160]}")
    log.warn("soil pH unavailable; using neutral pH 5.8 everywhere")
    return np.full(grid.shape, np.nan, dtype=np.float32)


# --------------------------------------------------------------------------- tree species (France)
def _norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return s


def classify_forest(essence, tfv):
    """Map BD Foret 'essence' / 'tfv' text to one of FOREST_TYPES (1-based index) or 0."""
    e, t = _norm(essence), _norm(tfv)
    if "lande" in t or "herbac" in t:
        return 0
    txt = e if e and e not in ("nc", "nr") else t
    if "conifer" in txt and "feuillu" in txt:  # conifer/broadleaf mixtures
        return FOREST_TYPES.index("mixed") + 1
    rules = [  # order matters: "sapin" and "autre que pin" contain "pin"
        ("autre que pin", "sprucefir"), ("sapin", "sprucefir"), ("epicea", "sprucefir"),
        ("douglas", "sprucefir"), ("meleze", "sprucefir"),
        ("hetre", "beech"), ("chene", "oak"), ("chataign", "chestnut"), ("pin", "pine"),
        ("mixte", "mixed"), ("melange de feuillus", "mixedbroad"), ("conifer", "sprucefir"),
        ("robinier", "otherbroad"), ("peupl", "otherbroad"), ("feuillu", "mixedbroad"),
    ]
    for key, cat in rules:
        if key in txt:
            return FOREST_TYPES.index(cat) + 1
    if "foret" in t:
        return FOREST_TYPES.index("mixedbroad") + 1
    return 0


def bdforet(grid: Grid):
    """Rasterise IGN BD Foret V2 polygons into forest-type fractions."""
    from rasterio.features import rasterize

    fine = grid.finer(SUB)
    minx, miny, maxx, maxy = grid.bounds_3857
    url = "https://data.geopf.fr/wfs/ows"
    layer = "LANDCOVER.FORESTINVENTORY.V2:formation_vegetale"
    raster = np.zeros(fine.shape, dtype=np.uint8)
    step = 40000.0  # 40 km query tiles
    n_feat, seen_ess = 0, {}
    xs = np.arange(minx, maxx, step)
    ys = np.arange(miny, maxy, step)
    t0 = time.time()
    for x in xs:
        for y in ys:
            bb = (x, y, min(x + step, maxx), min(y + step, maxy))
            start = 0
            while True:
                params = {
                    "SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature",
                    "TYPENAMES": layer, "OUTPUTFORMAT": "application/json",
                    "SRSNAME": "EPSG:3857", "BBOX": f"{bb[0]},{bb[1]},{bb[2]},{bb[3]},EPSG:3857",
                    "COUNT": 5000, "STARTINDEX": start,
                }
                data = None
                for attempt in range(4):
                    try:
                        r = requests.get(url, params=params, headers=UA, timeout=240)
                        r.raise_for_status()
                        data = r.json()
                        break
                    except Exception as exc:
                        if attempt == 3:
                            log.warn(f"BD Foret tile skipped: {exc.__class__.__name__}: {str(exc)[:200]}")
                        else:
                            time.sleep(5 * (attempt + 1))
                if data is None:
                    break
                feats = data.get("features", [])
                shapes = []
                for f in feats:
                    p = f.get("properties") or {}
                    cat = classify_forest(p.get("essence"), p.get("tfv"))
                    ess = p.get("essence") or p.get("tfv") or "?"
                    seen_ess[ess] = seen_ess.get(ess, 0) + 1
                    if cat and f.get("geometry"):
                        shapes.append((f["geometry"], cat))
                if shapes:
                    _burn(raster, shapes, fine.transform, rasterize)
                n_feat += len(feats)
                if len(feats) < 5000:
                    break
                start += 5000
    if n_feat == 0:
        raise RuntimeError("no BD Foret polygons returned")
    out = {name: block_mean((raster == i + 1).astype(np.float32), SUB).astype(np.float16)
           for i, name in enumerate(FOREST_TYPES)}
    top = sorted(seen_ess.items(), key=lambda kv: -kv[1])[:25]
    log.info(f"BD Foret: {n_feat} polygons in {time.time() - t0:.0f}s; essences: "
             + "; ".join(f"{k} ({v})" for k, v in top))
    log.info("forest types: " + ", ".join(f"{k} {float(v.mean()) * 100:.1f}%" for k, v in out.items()))
    return out


def _burn(raster, shapes, transform, rasterize):
    burned = rasterize(shapes, out_shape=raster.shape, transform=transform, fill=0, dtype=np.uint8)
    np.copyto(raster, burned, where=burned > 0)


# --------------------------------------------------------------------------- build all
def build(region, grid: Grid, cache_path, workdir):
    bbox = region["bbox"]
    layers = {}
    try:
        layers["elev"] = elevation(bbox, grid)
    except Exception as exc:
        log.error("elevation failed", exc)
        layers["elev"] = np.full(grid.shape, np.nan, np.float32)
    try:
        for k, v in landcover(bbox, grid).items():
            layers["lc_" + k] = v
    except Exception as exc:
        log.error("landcover failed", exc)
    try:
        layers["ph"] = soil_ph(bbox, grid, workdir)
    except Exception as exc:
        log.error("soil pH failed", exc)
    if region.get("bdforet"):
        try:
            for k, v in bdforet(grid).items():
                layers["ft_" + k] = v
        except Exception as exc:
            log.error("BD Foret failed (falling back to generic forest)", exc)
    np.savez_compressed(cache_path, **layers)
    log.info(f"static layers cached: {sorted(layers)}")
    return layers
