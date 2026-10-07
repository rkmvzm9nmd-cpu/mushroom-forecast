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


def wfs_features(layer, grid: Grid, label, cql=None, page=5000, step=40000.0):
    """Yield GeoJSON features (EPSG:3857) from the Geoplateforme WFS over the grid,
    in 40 km tiles with paging. If a CQL filter is given but rejected, falls back
    to a plain bbox query (filter then has to be applied by the caller)."""
    url = "https://data.geopf.fr/wfs/ows"
    minx, miny, maxx, maxy = grid.bounds_3857
    use_cql = cql is not None
    for x in np.arange(minx, maxx, step):
        for y in np.arange(miny, maxy, step):
            bb = (x, y, min(x + step, maxx), min(y + step, maxy))
            start = 0
            while True:
                params = {"SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature",
                          "TYPENAMES": layer, "OUTPUTFORMAT": "application/json",
                          "SRSNAME": "EPSG:3857", "COUNT": page, "STARTINDEX": start}
                bbox_txt = f"{bb[0]},{bb[1]},{bb[2]},{bb[3]}"
                if use_cql:
                    params["CQL_FILTER"] = f"BBOX(geom,{bbox_txt},'EPSG:3857') AND ({cql})"
                else:
                    params["BBOX"] = bbox_txt + ",EPSG:3857"
                data = None
                for attempt in range(4):
                    try:
                        r = requests.get(url, params=params, headers=UA, timeout=240)
                        if r.status_code == 400 and use_cql:
                            log.warn(f"{label}: server rejected filter ({r.text[:150]}); using plain bbox")
                            use_cql = False
                            break
                        r.raise_for_status()
                        data = r.json()
                        break
                    except Exception as exc:
                        if attempt == 3:
                            log.warn(f"{label} tile skipped: {exc.__class__.__name__}: {str(exc)[:200]}")
                        else:
                            time.sleep(5 * (attempt + 1))
                if data is None:
                    if not use_cql and "CQL_FILTER" in params:
                        continue  # retry this tile without the filter
                    break
                feats = data.get("features", [])
                yield from feats
                if len(feats) < page:
                    break
                start += page


def _burn(raster, shapes, transform):
    from rasterio.features import rasterize
    burned = rasterize(shapes, out_shape=raster.shape, transform=transform, fill=0, dtype=np.uint8)
    np.copyto(raster, burned, where=burned > 0)


def _rasterise_features(feats, grid: Grid, classify, label):
    fine = grid.finer(SUB)
    raster = np.zeros(fine.shape, dtype=np.uint8)
    seen, shapes, n = {}, [], 0
    t0 = time.time()
    for f in feats:
        n += 1
        p = f.get("properties") or {}
        cat, key = classify(p)
        seen[key] = seen.get(key, 0) + 1
        if cat and f.get("geometry"):
            shapes.append((f["geometry"], cat))
        if len(shapes) >= 5000:
            _burn(raster, shapes, fine.transform)
            shapes = []
    if shapes:
        _burn(raster, shapes, fine.transform)
    if n == 0:
        raise RuntimeError(f"no {label} polygons returned")
    top = sorted(seen.items(), key=lambda kv: -kv[1])[:25]
    log.info(f"{label}: {n} polygons in {time.time() - t0:.0f}s; classes: "
             + "; ".join(f"{k} ({v})" for k, v in top))
    return raster


def bdforet(grid: Grid):
    """IGN BD Foret V2 polygons -> forest-type fractions."""
    feats = wfs_features("LANDCOVER.FORESTINVENTORY.V2:formation_vegetale", grid, "BD Foret")
    raster = _rasterise_features(
        feats, grid, lambda p: (classify_forest(p.get("essence"), p.get("tfv")),
                                p.get("essence") or p.get("tfv") or "?"), "BD Foret")
    out = {"ft_" + name: block_mean((raster == i + 1).astype(np.float32), SUB).astype(np.float16)
           for i, name in enumerate(FOREST_TYPES)}
    log.info("forest types: " + ", ".join(f"{k[3:]} {float(v.mean()) * 100:.1f}%" for k, v in out.items()))
    return out


# --------------------------------------------------------------------------- grassland (France: RPG)
PASTURE_TYPES = ["permanent", "rough", "temporary"]
# RPG crop groups that are ploughed / cultivated land (arable, fallow, annual forage),
# plus orchards and vines: none of these is old undisturbed grassland.
TILLED_GROUPS = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "11", "14", "15", "16",
                 "20", "21", "22", "23", "24", "25", "26"}


def classify_pasture(p):
    """RPG declared parcel -> 1 permanent pasture, 2 rough grazing / summer pasture,
    3 temporary (re-sown) grassland, 4 tilled or cultivated land, 0 other."""
    group = str(p.get("code_group") or "")
    code = str(p.get("code_cultu") or "").upper()
    if code in ("PPH", "SPH") or group == "18":
        return 1, f"permanent ({code})"
    if code in ("SPL", "BOP") or group == "17":
        return 2, f"rough/estive ({code})"
    if group == "19" or code in ("PTR", "PRL"):
        return 3, f"temporary ({code})"
    if group in TILLED_GROUPS:
        return 4, f"tilled/cultivated group {group}"
    return 0, f"other group {group}"


def _with_organic(p):
    cat, key = classify_pasture(p)
    if cat in (1, 2, 3) and p.get("bio") in (True, "true", "True", 1, "1"):
        return cat | 8, key + " organic"
    return cat, key


def rpg(grid: Grid):
    # all declared parcels: grassland types plus ploughed land, so arable fields the
    # satellite mistakes for grass (young cereals) can be excluded. Bit 8 = organic farm.
    feats = wfs_features("RPG.LATEST:parcelles_graphiques", grid, "RPG")
    raster = _rasterise_features(feats, grid, _with_organic, "RPG")
    base = raster & 7
    out = {"pa_" + name: block_mean((base == i + 1).astype(np.float32), SUB).astype(np.float16)
           for i, name in enumerate(PASTURE_TYPES)}
    out["pa_tilled"] = block_mean((base == 4).astype(np.float32), SUB).astype(np.float16)
    out["pa_organic"] = block_mean((raster >= 8).astype(np.float32), SUB).astype(np.float16)
    log.info("parcel types: " + ", ".join(f"{k[3:]} {float(v.mean()) * 100:.1f}%" for k, v in out.items()))
    return out


HISTORY_YEARS = (2015, 2016, 2017)


def rpg_history(grid: Grid):
    """Where permanent pasture / rough grazing was already declared in the oldest
    available register (2015+). Grass that was permanent then and still is now has
    most likely not been ploughed for a decade."""
    for year in HISTORY_YEARS:
        try:
            feats = wfs_features(f"RPG.{year}:parcelles_graphiques", grid, f"RPG {year}",
                                 cql="code_group IN ('17','18')")
            raster = _rasterise_features(
                feats, grid, lambda p: ((1, "permanent then") if str(p.get("code_group")) in ("17", "18")
                                        else (0, "other")), f"RPG {year}")
            frac = block_mean((raster == 1).astype(np.float32), SUB).astype(np.float16)
            log.info(f"RPG {year}: permanent grass then covers {float(frac.mean()) * 100:.1f}% of region")
            return {"pa_hist_perm": frac, "pa_hist_year": np.array(year)}
        except Exception as exc:
            log.warn(f"RPG {year} unavailable: {exc.__class__.__name__}: {str(exc)[:150]}")
    raise RuntimeError("no historical RPG year available")


# --------------------------------------------------------------------------- French soil pH (GIS Sol / INRAE)
GSN_DOI = "doi:10.57745/3QFT2T"   # French maps for the Global Soil Nutrient map (pH, 250 m, Etalab 2.0)


def french_ph(grid: Grid, workdir):
    from rasterio.enums import Resampling
    base = "https://entrepot.recherche.data.gouv.fr"
    r = requests.get(f"{base}/api/datasets/:persistentId/versions/:latest/files",
                     params={"persistentId": GSN_DOI}, headers=UA, timeout=60)
    r.raise_for_status()
    files = [(f["dataFile"]["filename"], f["dataFile"]["id"]) for f in r.json()["data"]]
    log.info("French soil maps available: " + ", ".join(n for n, _ in files))

    def is_ph(n):
        n = n.lower()
        return "ph" in n and n.endswith((".tif", ".tiff")) and not any(
            t in n for t in ("sd", "std", "unc", "_q05", "_q95", "kcl"))
    cands = [f for f in files if is_ph(f[0])]
    if not cands:
        raise RuntimeError("no pH file found in dataset")
    name, fid = cands[0]
    path = os.path.join(workdir, "fr_ph.tif")
    with requests.get(f"{base}/api/access/datafile/{fid}", headers=UA, timeout=600, stream=True) as d:
        d.raise_for_status()
        with open(path, "wb") as f:
            for chunk in d.iter_content(1 << 20):
                f.write(chunk)
    ph, _ = _reproject_tiles([path], grid, Resampling.bilinear, np.float32, -9999.0)
    ph[(ph <= 0) | (ph > 140)] = np.nan
    if np.nanmedian(ph) > 14:
        ph = ph / 10.0
    log.info(f"French pH map ({name}): covers {np.isfinite(ph).mean() * 100:.0f}% of grid, "
             f"range {np.nanmin(ph):.1f}-{np.nanmax(ph):.1f}")
    return ph


def build_ph(region, grid, workdir):
    soil = soil_ph(region["bbox"], grid, workdir)  # SoilGrids, worldwide
    if region.get("country") == "FR":
        try:
            fr = french_ph(grid, workdir)
            both = np.isfinite(fr) & np.isfinite(soil)
            if both.any():
                log.info(f"pH France vs SoilGrids: mean {np.nanmean(fr[both]):.2f} vs {np.nanmean(soil[both]):.2f}, "
                         f"correlation {np.corrcoef(fr[both], soil[both])[0, 1]:.2f}")
            return {"ph": np.where(np.isfinite(fr), fr, soil).astype(np.float32), "ph_src_fr": np.isfinite(fr)}
        except Exception as exc:
            log.warn(f"French pH map unavailable, using SoilGrids: {exc.__class__.__name__}: {str(exc)[:200]}")
    return {"ph": soil}


# --------------------------------------------------------------------------- cached groups
# Bump a version to rebuild just that group on the next run.
GROUPS = {
    "dem": (1, lambda r, g, w: {"elev": elevation(r["bbox"], g)}),
    "lc": (1, lambda r, g, w: {"lc_" + k: v for k, v in landcover(r["bbox"], g).items()}),
    "ph": (2, build_ph),
    "bdforet": (1, lambda r, g, w: bdforet(g)),
    "rpg": (3, lambda r, g, w: rpg(g)),
    "rpg_hist": (1, lambda r, g, w: rpg_history(g)),
}
FRANCE_ONLY = {"bdforet", "rpg", "rpg_hist"}


def _migrate_v1(region, cache_dir, rdir):
    """Split the original single-file cache into per-group files (saves refetching)."""
    old = os.path.join(cache_dir, f"{region['id']}.npz")
    if not os.path.exists(old):
        return
    with np.load(old) as z:
        layers = {k: z[k] for k in z.files}
    shape = layers["elev"].shape
    split = {"dem": ["elev"], "lc": [k for k in layers if k.startswith("lc_")],
             "bdforet": [k for k in layers if k.startswith("ft_")]}
    for g, keys in split.items():
        if keys and not os.path.exists(os.path.join(rdir, g + ".npz")):
            np.savez_compressed(os.path.join(rdir, g + ".npz"), _v=GROUPS[g][0], _shape=shape,
                                **{k: layers[k] for k in keys})
    os.remove(old)
    log.info("static cache migrated to per-layer files")


def load_all(region, grid: Grid, cache_dir):
    rdir = os.path.join(cache_dir, region["id"])
    os.makedirs(rdir, exist_ok=True)
    _migrate_v1(region, cache_dir, rdir)
    layers = {}
    france = region.get("country") == "FR"
    for g, (ver, fn) in GROUPS.items():
        if g in FRANCE_ONLY and not france:
            continue
        path = os.path.join(rdir, g + ".npz")
        if os.path.exists(path):
            with np.load(path) as z:
                if int(z["_v"]) == ver and tuple(z["_shape"]) == grid.shape:
                    layers.update({k: z[k] for k in z.files if not k.startswith("_")})
                    continue
        log.info(f"{region['id']}: building layer group '{g}'")
        try:
            got = fn(region, grid, rdir)
            np.savez_compressed(path, _v=ver, _shape=grid.shape, **got)
            layers.update(got)
        except Exception as exc:
            log.error(f"layer group '{g}' failed (will retry next run)", exc)
    if "elev" not in layers:
        layers["elev"] = np.full(grid.shape, np.nan, np.float32)
    return layers
