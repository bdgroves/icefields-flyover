"""Step 2: prepare the imagery for rendering.

  sentinel2_*_graded.png  a little more saturation and contrast, offsetting the
                          renderer's sun/sky tint and tone mapping. On the steepest
                          faces the satellite view (straight down, mid-morning) is
                          smeared and often in shadow, so faces steeper than ~50
                          degrees lean toward the grey of Rockies limestone and the
                          renderer's evening sun lights them. Ice and snow are left
                          alone. Lakes with a ridge's shadow or haze lying on the water
                          are repainted in their own turquoise.

    pixi run prep
"""
from __future__ import annotations

import sys

import numpy as np
import rasterio
from PIL import Image
from scipy import ndimage

import config

SATURATION = 1.18
LIMESTONE = np.array([0.66, 0.65, 0.63], dtype=np.float32)
CLIFF_DEG = (45.0, 65.0)


def slope_deg() -> np.ndarray:
    with rasterio.open(config.DEM) as src:
        z = ndimage.gaussian_filter(src.read(1).astype(np.float32), 0.8)
    gy, gx = np.gradient(z, config.DEM_RES)
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)


def cliffs(rgb: np.ndarray, left: float, top: float, slope: np.ndarray) -> np.ndarray:
    gl, gt, _, _ = config.grid()
    res = config.IMAGE_RES
    h, w = rgb.shape[:2]
    rows = (gt - (top - (np.arange(h) + 0.5) * res)) / config.DEM_RES - 0.5
    cols = (left + (np.arange(w) + 0.5) * res - gl) / config.DEM_RES - 0.5
    lo, hi = CLIFF_DEG
    k = np.empty((h, w), dtype=np.float32)
    for r0 in range(0, h, 512):
        RR, CC = np.meshgrid(rows[r0:r0 + 512], cols, indexing="ij")
        t = np.clip((ndimage.map_coordinates(slope, [RR, CC], order=1, mode="nearest") - lo) / (hi - lo), 0, 1)
        k[r0:r0 + 512] = t * t * (3 - 2 * t)
    lum = rgb.mean(axis=-1, keepdims=True)
    snow = np.clip((lum[..., 0] - 0.62) / 0.15, 0, 1)            # leave ice and snow as they are
    k *= 0.75 * (1 - snow)
    local = ndimage.uniform_filter(lum[..., 0], size=15)[..., None] + 1e-3
    rock = LIMESTONE * np.clip(0.88 + 0.3 * (lum / local - 1.0), 0.7, 1.15)
    return rgb * (1 - k[..., None]) + rock * k[..., None]


def to_image(grid: np.ndarray, left: float, top: float, h: int, w: int, order: int = 1) -> np.ndarray:
    """Sample a DEM-grid array at every image pixel."""
    gl, gt, _, _ = config.grid()
    res = config.IMAGE_RES
    rows = (gt - (top - (np.arange(h) + 0.5) * res)) / config.DEM_RES - 0.5
    cols = (left + (np.arange(w) + 0.5) * res - gl) / config.DEM_RES - 0.5
    out = np.empty((h, w), dtype=np.float32)
    for r0 in range(0, h, 512):
        RR, CC = np.meshgrid(rows[r0:r0 + 512], cols, indexing="ij")
        out[r0:r0 + 512] = ndimage.map_coordinates(grid.astype(np.float32), [RR, CC], order=order, mode="nearest")
    return out


def lakes(rgb: np.ndarray, left: float, top: float, z: np.ndarray, slope: np.ndarray) -> np.ndarray:
    """Clean up each lake. The satellite passed mid-morning, so some lakes have a ridge's
    shadow (or a little haze) lying across the water. Start from each patch of clearly
    turquoise water, find its level from the DEM, and repaint the dark, bluish pixels
    joined to it at that same level with the lake's own median colour."""
    h, w = rgb.shape[:2]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    clear = (b > r + 0.06) & (g > r + 0.04) & (b > 0.25)
    turquoise = (r < 0.2) & (b > r + 0.15) & (g > r + 0.15)        # glacial lake water, not blue ice
    seeds, n = ndimage.label(turquoise)
    sizes = ndimage.sum(turquoise, seeds, np.arange(1, n + 1))
    done = 0
    for i, sl in enumerate(ndimage.find_objects(seeds), start=1):
        if sl is None or sizes[i - 1] < 300:                       # under ~4 ha: skip
            continue
        pad = 60
        rs = slice(max(sl[0].start - pad, 0), min(sl[0].stop + pad, h))
        cs = slice(max(sl[1].start - pad, 0), min(sl[1].stop + pad, w))
        hh, ww = rs.stop - rs.start, cs.stop - cs.start
        top_, left_ = top - rs.start * config.IMAGE_RES, left + cs.start * config.IMAGE_RES
        zz = to_image(z, left_, top_, hh, ww)
        ss = to_image(slope, left_, top_, hh, ww)
        sub = rgb[rs, cs]                                          # a view: edits land in rgb
        seed = seeds[rs, cs] == i
        level = float(np.median(zz[seed]))
        sr, sg, sb = sub[..., 0], sub[..., 1], sub[..., 2]
        water_ish = (sb > sr) & (sb >= sg - 0.02) & (sub.mean(axis=-1) < 0.45)
        cand = (np.abs(zz - level) < 2.5) & (ss < 2.0) & (water_ish | seed)
        lab, _ = ndimage.label(cand)
        ids = np.unique(lab[seed])
        lake = np.isin(lab, ids[ids > 0])
        lum = sub.mean(axis=-1)
        bright = seed & (lum >= np.percentile(lum[seed], 40))         # sunlit water
        colour = np.median(sub[bright], axis=0)
        ref = float(colour.mean())
        fix = lake & (~clear[rs, cs] | (lum < 0.8 * ref))               # off-colour or in shadow
        level_ok = (np.abs(zz - level) < 2.5) & (ss < 2.0) & (lum < 0.45)
        fix |= ndimage.binary_dilation(fix, iterations=2) & level_ok & (lum < 0.9 * ref)   # close the seams
        if fix.sum() < 20:
            continue
        var = np.clip(1 + 0.15 * (lum - lum[fix].mean()), 0.93, 1.07)
        sub[fix] = colour[None, :] * var[fix][:, None]
        done += 1
    print(f"  {done} lakes cleaned up")
    return rgb


def grade(path, left, top, z, slope) -> None:
    Image.MAX_IMAGE_PIXELS = None
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    rgb = lakes(rgb, left, top, z, slope)
    rgb = cliffs(rgb, left, top, slope)
    lum = (rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))[..., None]
    rgb = np.clip(lum + (rgb - lum) * SATURATION, 0.0, 1.0)
    rgb += 0.12 * (rgb - 0.5) * (1.0 - np.abs(2.0 * rgb - 1.0))     # gentle S-curve
    Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)).save(config.graded(path))
    print(f"wrote {config.graded(path).name}")


def main() -> int:
    tiles = config.image_tiles()
    missing = [p.name for p in [config.DEM] + [t[0] for t in tiles] if not p.exists()]
    if missing:
        print("Missing", ", ".join(missing), "- run: pixi run data")
        return 1
    slope = slope_deg()
    with rasterio.open(config.DEM) as src:
        z = src.read(1).astype(np.float32)
    for path, left, top, _, _ in tiles:
        grade(path, left, top, z, slope)
    return 0


if __name__ == "__main__":
    sys.exit(main())
