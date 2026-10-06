"""Shared settings for the Icefields Parkway flyover."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cache"
OUT = ROOT / "out"

CRS = "EPSG:32611"             # WGS 84 / UTM zone 11N, metres (Sentinel-2's own grid here)

# Lake Louise to the Columbia Icefield, with room either side for the horizon.
WEST, EAST = -117.45, -116.05
SOUTH, NORTH = 51.33, 52.30
DEM_RES = 25.0                 # metres; the source is the 30 m Copernicus DEM
IMAGE_RES = 12.0               # Sentinel-2 is 10 m
IMAGE_TILES = 2                # the imagery is split north/south to keep each texture a sensible size

DEM = DATA / "icefields_dem_25m.tif"
IMAGE_BEST = "2025-09-25"                       # a nearly cloud-free pass, golden larch season
IMAGE_DATE = ("2025-09-12", "2025-09-30")       # then other late-September passes
IMAGE_FALLBACK = ("2025-08-01", "2025-09-30")   # to fill any cloudy gaps


def image_path(k: int) -> Path:
    return DATA / f"sentinel2_{k}.png"


def graded(path: Path) -> Path:
    return path.with_name(path.stem + "_graded.png")


# Evening light: September 25, 2025, 5:45 pm MDT. The sun is low in the west-south-west.
SUN_UTC = (2025, 9, 25, 23, 45, 0)
SUN_LATLON = (51.8, -116.6)

VIDEO = OUT / "icefields_parkway_flyover.mp4"


def grid():
    """(left, top, width px, height px) of the DEM grid in CRS metres."""
    from rasterio.warp import transform_bounds
    l, b, r, t = transform_bounds("EPSG:4326", CRS, WEST, SOUTH, EAST, NORTH)
    left, top = round(l / 100) * 100, round(t / 100) * 100
    w, h = int((r - left) // DEM_RES), int((top - b) // DEM_RES)
    return left, top, w, h


def bounds():
    left, top, w, h = grid()
    return left, top, left + w * DEM_RES, top - h * DEM_RES


def image_tiles():
    """(path, left, top, right, bottom) for each imagery tile, north first."""
    left, top, right, bottom = bounds()
    edges = [top - (top - bottom) * k / IMAGE_TILES for k in range(IMAGE_TILES + 1)]
    edges = [round(e / IMAGE_RES) * IMAGE_RES for e in edges]
    edges[0], edges[-1] = top, bottom
    return [(image_path(k), left, edges[k], right, edges[k + 1]) for k in range(IMAGE_TILES)]

LEFT, TOP, RIGHT, BOTTOM = bounds()
