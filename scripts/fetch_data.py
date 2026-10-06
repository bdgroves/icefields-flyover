"""Step 1: download the terrain and imagery for the Icefields Parkway flyover into data/.

  icefields_dem_25m.tif   Copernicus GLO-30 elevation (ESA, free), the 1-degree tiles
                          covering Lake Louise to the Columbia Icefield, reprojected
                          to a 25 m UTM grid
  sentinel2_*.png         Sentinel-2 true colour at 12 m, from the nearly
                          cloud-free pass of September 25, 2025; any cloud or shadow
                          is filled from the next-clearest passes

Both come from public AWS buckets (no account needed); files that already exist are
skipped, so an interrupted run can simply be restarted.

    pixi run data
"""
from __future__ import annotations

import json
import math
import sys
import time
import urllib.request

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from scipy import ndimage

import config

COP = "https://copernicus-dem-30m.s3.amazonaws.com"
STAC = "https://earth-search.aws.element84.com/v1/search"
# Sentinel-2 scene classes to keep: vegetation, bare, water, unclassified, snow/ice
# (dropping no-data, saturated, dark/cloud shadow, cloud and cirrus)
GOOD_SCL = {2, 4, 5, 6, 7, 11}
GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES", VSI_CACHE="TRUE")


def cop_url(lat: int, lon: int) -> str:
    ns = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
    ew = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
    name = f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"
    return f"{COP}/{name}/{name}.tif"


def build_dem() -> None:
    if config.DEM.exists():
        print(f"Terrain already present: {config.DEM.name}")
        return
    left, top, w, h = config.grid()
    tf = from_origin(left, top, config.DEM_RES, config.DEM_RES)
    dem = np.full((h, w), np.nan, dtype=np.float32)
    tiles = [(la, lo) for la in range(math.floor(config.SOUTH), math.floor(config.NORTH) + 1)
             for lo in range(math.floor(config.WEST), math.floor(config.EAST) + 1)]
    print(f"Copernicus DEM: {len(tiles)} tiles -> {w} x {h} at {config.DEM_RES:g} m")
    with rasterio.Env(**GDAL_ENV):
        for la, lo in tiles:
            url = cop_url(la, lo)
            print(f"  {url.rsplit('/', 1)[-1]}")
            part = np.full_like(dem, np.nan)
            with rasterio.open("/vsicurl/" + url) as src:
                reproject(rasterio.band(src, 1), part, dst_transform=tf, dst_crs=config.CRS,
                          src_nodata=src.nodata, dst_nodata=np.nan, resampling=Resampling.bilinear)
            fill = np.isnan(dem) & np.isfinite(part)
            dem[fill] = part[fill]
    if np.isnan(dem).any():
        print(f"  filling {np.isnan(dem).mean():.2%} gaps from neighbours")
        idx = ndimage.distance_transform_edt(np.isnan(dem), return_distances=False, return_indices=True)
        dem = dem[idx[0], idx[1]]
    prof = {"driver": "GTiff", "width": w, "height": h, "count": 1, "dtype": "float32", "crs": config.CRS,
            "transform": tf, "nodata": None, "compress": "deflate", "tiled": True}
    tmp = config.DEM.with_suffix(".part.tif")
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(dem, 1)
    tmp.replace(config.DEM)
    print(f"Saved {config.DEM.name}: {dem.min():,.0f}-{dem.max():,.0f} m")


def stac_search(dates) -> list[dict]:
    body = {"collections": ["sentinel-2-l2a"], "bbox": [config.WEST, config.SOUTH, config.EAST, config.NORTH],
            "datetime": f"{dates[0]}T00:00:00Z/{dates[1]}T23:59:59Z", "limit": 200,
            "query": {"eo:cloud_cover": {"lt": 40}}}
    req = urllib.request.Request(STAC, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    for k in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)["features"]
        except Exception as e:                       # noqa: BLE001
            print(f"  search attempt {k + 1}: {e}")
            time.sleep(4 * (k + 1))
    raise RuntimeError("Sentinel-2 search failed")


def scenes() -> list[dict]:
    """Passes to use, best first: the main window by cloud cover, then the fallback window."""
    out, seen = [], set()
    for dates in (config.IMAGE_DATE, config.IMAGE_FALLBACK):
        feats = sorted(stac_search(dates), key=lambda f: (not f["properties"]["datetime"].startswith(config.IMAGE_BEST),
                                                          f["properties"].get("eo:cloud_cover", 100)))
        for f in feats:
            if f["id"] not in seen and "visual" in f["assets"] and "scl" in f["assets"]:
                seen.add(f["id"])
                out.append(f)
    return out


def build_image(path, left, top, right, bottom, items) -> None:
    if path.exists():
        print(f"Imagery already present: {path.name}")
        return
    res = config.IMAGE_RES
    w, h = round((right - left) / res), round((top - bottom) / res)
    tf = from_origin(left, top, res, res)
    rgb = np.zeros((3, h, w), dtype=np.uint8)
    have = np.zeros((h, w), dtype=bool)
    print(f"{path.name}: {w} x {h} px at {res:g} m")
    with rasterio.Env(**GDAL_ENV):
        for f in items:
            p = f["properties"]
            # skip passes whose footprint misses this tile
            from shapely.geometry import box, shape
            from rasterio.warp import transform_geom
            foot = shape(transform_geom("EPSG:4326", config.CRS, f["geometry"]))
            if not foot.intersects(box(left, bottom, right, top)):
                continue
            scl = np.zeros((h, w), dtype=np.uint8)
            with rasterio.open("/vsicurl/" + f["assets"]["scl"]["href"]) as src:
                reproject(rasterio.band(src, 1), scl, dst_transform=tf, dst_crs=config.CRS,
                          src_nodata=0, dst_nodata=0, resampling=Resampling.nearest)
            good = np.isin(scl, list(GOOD_SCL)) & ~have
            good = ndimage.binary_erosion(good, iterations=2)          # stay clear of cloud edges
            if good.sum() < 0.002 * w * h:
                continue
            part = np.zeros((3, h, w), dtype=np.uint8)
            with rasterio.open("/vsicurl/" + f["assets"]["visual"]["href"]) as src:
                for b in range(3):
                    reproject(rasterio.band(src, b + 1), part[b], dst_transform=tf, dst_crs=config.CRS,
                              src_nodata=0, dst_nodata=0, resampling=Resampling.average)
            good &= part.max(axis=0) > 0
            rgb[:, good] = part[:, good]
            have |= good
            print(f"  {p.get('datetime', '')[:10]} {f['id'][:24]:<24} cloud {p.get('eo:cloud_cover', 0):4.1f}%"
                  f"  -> {have.mean():6.1%} filled")
            if have.mean() > 0.9995:
                break
    if not have.all():
        print(f"  {1 - have.mean():.2%} left without a clear view: filled from neighbouring pixels")
        idx = ndimage.distance_transform_edt(~have, return_distances=False, return_indices=True)
        rgb = rgb[:, idx[0], idx[1]]
    tmp = path.with_suffix(".part.png")
    Image.fromarray(np.moveaxis(rgb, 0, -1)).save(tmp)
    tmp.replace(path)
    print(f"Saved {path.name}")


def main() -> int:
    config.CACHE.mkdir(parents=True, exist_ok=True)
    build_dem()
    tiles = config.image_tiles()
    if all(p.exists() for p, *_ in tiles):
        print("Imagery already present")
    else:
        items = scenes()
        print(f"Sentinel-2: {len(items)} passes to draw from, clearest first "
              f"({', '.join(sorted({f['properties']['datetime'][:10] for f in items[:6]}))} ...)")
        for t in tiles:
            build_image(*t, items)
    print("Data ready. Next: pixi run prep")
    return 0


if __name__ == "__main__":
    sys.exit(main())
