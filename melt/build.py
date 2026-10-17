"""Build everything the melt film needs (runs on GitHub Actions; needs the network).

Writes melt/data/:
  dem.tif           today's ground and ice surface, 10 m, from Canada's 30 m MRDEM (cubic)
  bed.tif           the ground under the ice: surface minus the consensus ice thickness
                    (Farinotti et al. 2019), smoothed; off the ice it equals dem.tif
  glaciers.tif      which RGI 6 glacier each cell belongs to (index into place.json "glaciers")
  ice_now.tif       today's ice: the RGI outline trimmed to what Sentinel-2 still sees as snow/ice
  s1844.tif         the reconstructed 1844 surface of the Athabasca (see below)
  along.tif         distance along the Athabasca's flowline from its 1844 front, per cell
  texture.png       Sentinel-2 true colour, late summer 2025
  place.json        grid, glaciers, flowline, toe positions, projections, checks

The 1844 surface. Gadd (2011): the ice "was once level with the tops of the lateral
moraines" and its front reached Icefield Centre. Along the tongue the moraine crests are
found in cross-sections of the terrain; the 1844 surface follows them, and near the front
it follows a perfectly plastic snout (Nye), so it thins to nothing at the 1844 front.

    python melt/build.py
"""
from __future__ import annotations

import io
import json
import re
import sys
import zipfile
from pathlib import Path

import httpx
import numpy as np
import rasterio
from PIL import Image
from rasterio.features import rasterize, shapes
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform, transform_bounds
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

http = httpx.Client(timeout=300, follow_redirects=True)
MRDEM = "https://canelevation-dem.s3.ca-central-1.amazonaws.com/mrdem-30/mrdem-30-dtm.tif"
RGI_ZIP = "https://cluster.klima.uni-bremen.de/~oggm/rgi/rgi62.zip"
THICK = ["https://cluster.klima.uni-bremen.de/~fmaussion/icevol/composite/RGI60-02/{id}_thickness.tif"]  # as oggm.shop.bedtopo
OGGM = ("https://cluster.klima.uni-bremen.de/~oggm/oggm-standard-projections/oggm_v16/2025.6/"
        "w5e5/per_glacier_spinup/CMIP6/2100/RGI02/")
STAC = "https://earth-search.aws.element84.com/v1/search"
GDAL = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF")
info: dict = {"checks": {}}


def say(s):
    print("= " + s, flush=True)


def grid():
    l, b, r, t = transform_bounds("EPSG:4326", C.CRS, *C.BBOX)
    l, t = np.floor(l / 100) * 100, np.ceil(t / 100) * 100
    w, h = int(np.ceil((r - l) / C.RES)), int(np.ceil((t - b) / C.RES))
    return from_origin(l, t, C.RES, C.RES), (h, w)


T, SHAPE = grid()
H, W = SHAPE
PROF = dict(driver="GTiff", width=W, height=H, count=1, crs=C.CRS, transform=T, compress="deflate",
            tiled=True, blockxsize=256, blockysize=256)


def xy(lat, lon):
    x, y = transform("EPSG:4326", C.CRS, [lon], [lat])
    return x[0], y[0]


def rc(x, y):
    c, r = ~T * (x, y)
    return int(r), int(c)


def save(name, arr, dtype="float32", **kw):
    p = dict(PROF, dtype=dtype, **kw)
    if dtype == "float32":
        p.update(predictor=3)
    with rasterio.open(C.DATA / name, "w", **p) as d:
        d.write(arr.astype(dtype), 1)


def warp_to_grid(src_path_or_ds, resampling=Resampling.cubic, nodata=np.nan):
    out = np.full(SHAPE, np.nan, np.float32)
    with rasterio.Env(**GDAL):
        src = rasterio.open(src_path_or_ds) if isinstance(src_path_or_ds, str) else src_path_or_ds
        with src:
            reproject(rasterio.band(src, 1), out, dst_transform=T, dst_crs=C.CRS, resampling=resampling,
                      dst_nodata=nodata, src_nodata=src.nodata)
    return out


# -------------------------------------------------------------------- terrain
def terrain():
    dem = warp_to_grid("/vsicurl/" + MRDEM, Resampling.cubic)
    if np.isnan(dem).any():
        idx = ndimage.distance_transform_edt(np.isnan(dem), return_distances=False, return_indices=True)
        dem = dem[tuple(idx)]
    dem = ndimage.gaussian_filter(dem, 0.8)        # soften the 30 m cells a little
    say(f"terrain {W} x {H} at {C.RES:g} m from MRDEM 30 m: {dem.min():.0f}-{dem.max():.0f} m")
    return dem


# -------------------------------------------------------------------- glaciers
def glaciers():
    import shapefile
    from shapely.geometry import box, mapping, shape
    from shapely.ops import transform as stf
    from pyproj import Transformer
    z = zipfile.ZipFile(io.BytesIO(http.get(RGI_ZIP).content))
    inner = [n for n in z.namelist() if re.search(r"02_rgi6\d.*\.zip$", n)][0]
    z = zipfile.ZipFile(io.BytesIO(z.read(inner)))
    base = [n for n in z.namelist() if n.endswith(".shp")][0][:-4]
    r = shapefile.Reader(shp=io.BytesIO(z.read(base + ".shp")), dbf=io.BytesIO(z.read(base + ".dbf")),
                         shx=io.BytesIO(z.read(base + ".shx")))
    fields = [f[0] for f in r.fields[1:]]
    win = box(*C.BBOX)
    tr = Transformer.from_crs("EPSG:4326", C.CRS, always_xy=True).transform
    gl = []
    for sr in r.iterShapeRecords():
        g = shape(sr.shape.__geo_interface__)
        if g.intersects(win):
            rec = dict(zip(fields, sr.record))
            gl.append({"id": rec["RGIId"], "name": rec.get("Name") or "", "area_km2": rec["Area"],
                       "zmin": rec["Zmin"], "zmax": rec["Zmax"], "geom": stf(tr, g)})
    gl.sort(key=lambda g: -g["area_km2"])
    lab = rasterize([(mapping(g["geom"]), i + 1) for i, g in enumerate(gl)], out_shape=SHAPE, transform=T,
                    fill=0, dtype="int16")
    ath = next(i for i, g in enumerate(gl) if g["id"] == C.ATHABASCA)
    say(f"{len(gl)} RGI glaciers in the window; Athabasca {gl[ath]['area_km2']} km2 (RGI 6, outline c. 2005)")
    return gl, lab, ath + 1


def thickness(gl, dem):
    """Consensus thickness (Farinotti et al. 2019) per glacier, on the grid. Falls back to the
    shallow-ice estimate from surface slope where a glacier's file can't be had."""
    th = np.zeros(SHAPE, np.float32)
    got = 0
    for g in gl:
        for url in THICK:
            try:
                r = http.get(url.format(id=g["id"]))
                if r.status_code != 200:
                    continue
                with rasterio.MemoryFile(r.content) as mf, mf.open() as src:
                    a = np.zeros(SHAPE, np.float32)
                    reproject(rasterio.band(src, 1), a, dst_transform=T, dst_crs=C.CRS,
                              resampling=Resampling.bilinear, src_nodata=src.nodata, dst_nodata=0)
                th = np.maximum(th, np.nan_to_num(a))
                got += 1
                break
            except Exception as e:
                print(f"  thickness {g['id']}: {e!r}")
    info["thickness_source"] = f"consensus (Farinotti et al. 2019) for {got} of {len(gl)} glaciers"
    say(info["thickness_source"])
    return th, got


def sia_thickness(dem, mask):
    """Shallow-ice / plastic estimate H = tau / (rho g sin(slope)), slope over ~500 m."""
    s = ndimage.gaussian_filter(dem, 25)
    gy, gx = np.gradient(s, C.RES)
    sl = np.clip(np.hypot(gx, gy), 0.05, 1.0)
    h = C.TAU / (C.RHO * C.G * np.sin(np.arctan(sl)))
    edge = ndimage.distance_transform_edt(mask) * C.RES
    return np.where(mask, np.minimum(h, np.sqrt(2 * C.TAU / (C.RHO * C.G) * edge) * 3 + 5), 0).astype(np.float32)


# -------------------------------------------------------------------- imagery
def imagery():
    lb = C.BBOX
    q = {"collections": ["sentinel-2-l2a"], "bbox": list(lb), "datetime": "2025-08-10T00:00:00Z/2025-09-30T23:59:59Z",
         "query": {"eo:cloud_cover": {"lt": 15}}, "limit": 50}
    items = http.post(STAC, json=q).json()["features"]
    items.sort(key=lambda f: (f["properties"].get("eo:cloud_cover", 99), -int(f["properties"]["datetime"][5:7])))
    rgb = np.full((3,) + SHAPE, np.nan, np.float32)
    swir = np.full(SHAPE, np.nan, np.float32)
    nir = np.full(SHAPE, np.nan, np.float32)
    used = []
    for it in items[:8]:
        a = it["assets"]
        bands = []
        for k in ("red", "green", "blue"):
            bands.append(warp_to_grid("/vsicurl/" + a[k]["href"], Resampling.bilinear, 0))
        sw = warp_to_grid("/vsicurl/" + a["swir16"]["href"], Resampling.bilinear, 0)
        ni = warp_to_grid("/vsicurl/" + a["nir"]["href"], Resampling.bilinear, 0)
        new = np.isnan(rgb[0]) & (bands[0] > 0)
        if new.any():
            for k in range(3):
                rgb[k][new] = bands[k][new]
            swir[new] = sw[new]
            nir[new] = ni[new]
            used.append(it["properties"]["datetime"][:10])
        if not np.isnan(rgb[0]).any():
            break
    say(f"Sentinel-2 {used}: {100 * (~np.isnan(rgb[0])).mean():.1f}% covered")
    info["imagery"] = used
    for k in range(3):
        rgb[k] = np.nan_to_num(rgb[k], nan=np.nanmedian(rgb[k]))
    swir = np.nan_to_num(swir, nan=np.nanmedian(swir))
    nir = np.nan_to_num(nir, nan=np.nanmedian(nir))
    ndsi = (rgb[1] - swir) / (rgb[1] + swir + 1e-6)
    # water reads as snow to NDSI; NDWI (green vs near-infrared) picks it out
    water = ((rgb[1] - nir) / (rgb[1] + nir + 1e-6) > 0.15) & (rgb[1] < 2500)
    ndsi = np.where(water, -1.0, ndsi)
    say(f"open water: {water.sum() * C.RES ** 2 / 1e4:.0f} ha")
    x = np.moveaxis(rgb, 0, -1) / 10000.0
    lo, hi = np.percentile(x, 1), np.percentile(x, 99.7)
    x = np.clip((x - lo) / (hi - lo), 0, 1) ** (1 / 1.25)
    return (x * 255 + 0.5).astype(np.uint8), ndsi


# -------------------------------------------------------------------- flowline and 1844
def flowline(dem, ath_mask):
    """The tongue's centre line, found as the cheapest way from the 1844 front (Icefield
    Centre) up to the icefield, when every metre of height costs a lot."""
    from skimage.graph import route_through_array
    k = 3
    z = ndimage.zoom(ndimage.gaussian_filter(dem, 2), 1 / k, order=1)
    a_rc = rc(*xy(*C.ICEFIELD_CENTRE))
    hi = ath_mask & (dem > np.percentile(dem[ath_mask], 85))
    cy, cx = ndimage.center_of_mass(hi)
    ar, ac = a_rc[0] // k, a_rc[1] // k
    br, bc = int(cy) // k, int(cx) // k
    zmin = z[ar, ac]
    cost = 1 + ((z - zmin).clip(0) / 40.0) ** 2
    path, _ = route_through_array(cost, (ar, ac), (br, bc), fully_connected=True, geometric=True)
    p = np.array(path, float) * k + k / 2
    p = np.column_stack([ndimage.gaussian_filter1d(p[:, j], 6, mode="nearest") for j in range(2)])
    p = recentre(p, dem, ath_mask)
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))] * C.RES
    ss = np.arange(0, seg[-1], C.RES)
    pr = np.interp(ss, seg, p[:, 0]); pc = np.interp(ss, seg, p[:, 1])
    return np.column_stack([pr, pc]), ss


def recentre(p, dem, ice, half_m=900, passes=3):
    """Move each point to the middle of the glacier across the valley, or to the valley floor
    where there's no ice (the forefield), then smooth. The cheapest-path line hugs a margin."""
    zsm = ndimage.gaussian_filter(dem, 3)
    for _ in range(passes):
        d = np.gradient(p, axis=0)
        q = p.copy()
        offs = np.arange(-half_m, half_m + 1, C.RES) / C.RES
        for i in range(len(p)):
            t = d[i] / (np.hypot(*d[i]) + 1e-9)
            n = np.array([-t[1], t[0]])
            rr = p[i, 0] + n[0] * offs; cc = p[i, 1] + n[1] * offs
            ic = ndimage.map_coordinates(ice.astype(float), [rr, cc], order=0, mode="constant") > 0.5
            if ic.any():
                lab, _ = ndimage.label(ic)
                mid = len(offs) // 2
                js = np.flatnonzero(ic)
                k = lab[js[np.argmin(np.abs(js - mid))]]          # the ice run nearest the line
                jj = np.flatnonzero(lab == k)
                o = offs[(jj.min() + jj.max()) // 2]
            else:
                z = ndimage.map_coordinates(zsm, [rr, cc], order=1, mode="nearest")
                near = np.abs(offs) * C.RES < 350
                o = offs[near][np.argmin(z[near])]
            q[i] = p[i] + n * o * 0.7
        p = np.column_stack([ndimage.gaussian_filter1d(q[:, j], 5, mode="nearest") for j in range(2)])
    return p


def moraine_crests(dem, s_now_ice, pr, pc, ss, toe_s):
    """Cross-sections across the valley every 50 m: from the ice margin (or the valley floor
    in the forefield) outward, the first ridge within 450 m is the lateral moraine crest."""
    out = []
    d = np.gradient(np.column_stack([pr, pc]), axis=0)
    for i in range(0, len(ss), 5):
        if ss[i] > toe_s + 3000:
            break
        t = d[i] / (np.hypot(*d[i]) + 1e-9)
        nrm = np.array([-t[1], t[0]])
        offs = np.arange(-800, 801, C.RES) / C.RES
        rr = pr[i] + nrm[0] * offs; cc = pc[i] + nrm[1] * offs
        ok = (rr >= 0) & (rr < H - 1) & (cc >= 0) & (cc < W - 1)
        z = ndimage.map_coordinates(dem, [rr, cc], order=1, mode="nearest")
        ice = ndimage.map_coordinates(s_now_ice.astype(float), [rr, cc], order=0) > 0.5
        mid = len(offs) // 2
        crests = []
        zs = ndimage.gaussian_filter1d(z, 1.5)
        for side in (-1, 1):
            j = mid
            while 0 < j < len(offs) - 1 and ice[j]:      # walk out of the ice
                j += side
            start = j
            best, smax = None, 0.0
            for jj in range(start + side, start + side * 50, side):
                if not (1 <= jj < len(offs) - 1) or not ok[jj]:
                    break
                rise = zs[jj] - zs[start]
                slope = (zs[jj + side] - zs[jj - side]) / (2 * C.RES)      # outward rise per metre
                smax = max(smax, slope)
                # a ridge top, or a bench where the outward slope eases to under 40% of its
                # steepest so far, after climbing at least 15 m: the moraine crest
                if rise > 15 and (slope <= 0 or slope < 0.4 * smax):
                    best = zs[jj]
                    break
            if best is not None:
                crests.append(best)
        floor = z[mid]
        if crests:
            out.append((ss[i], min(crests), floor))
    return out


def reconstruct_1844(dem, bed, ice_now, ath_mask, gl_lab):
    pts, ss = flowline(dem, ath_mask)
    pr, pc = pts[:, 0], pts[:, 1]
    # today's toe: first flowline point on today's ice
    on = ndimage.map_coordinates(ice_now.astype(float) * ath_mask, [pr, pc], order=0) > 0.5
    toe_i = int(np.argmax(on))
    toe_s = float(ss[toe_i])
    say(f"flowline {ss[-1] / 1000:.1f} km; today's toe {toe_s:.0f} m up-valley from Icefield Centre "
        f"(Gadd 2011: 'nearly 2 km')")
    info["checks"]["toe_distance_from_1844_m"] = round(toe_s)
    s_now = dem  # the DEM surface is today's ice surface on the glacier
    zflow = ndimage.map_coordinates(s_now, [pr, pc], order=1)
    bflow = ndimage.map_coordinates(bed, [pr, pc], order=1)
    cr = moraine_crests(dem, ice_now & (gl_lab > 0), pr, pc, ss, toe_s) if C.DETECT_MORAINES else []
    K = np.sqrt(2 * C.TAU / (C.RHO * C.G))
    s1844 = zflow.copy()
    if cr:
        cs = np.array(cr)
        crest = np.interp(ss, cs[:, 0], ndimage.median_filter(cs[:, 1], 5), left=np.nan, right=np.nan)
        say(f"moraine crests found in {len(cr)} cross-sections; crest - today's surface at the toe: "
            f"{np.nanmedian(crest[max(0, toe_i - 10):toe_i + 10] - zflow[max(0, toe_i - 10):toe_i + 10]):.0f} m")
        info["moraine_sections"] = [[round(a), round(b, 1), round(c, 1)] for a, b, c in cr]
    else:
        crest = np.full_like(ss, np.nan)
        say(f"1844 surface from the published moraine height: {C.DZ_TOE_M:g} m above today's toe, "
            f"fading to {C.UP_MIN_M:g} m {C.UP_FADE_M / 1000:g} km up-glacier")
    # above the crest data, the thickening fades to 10 m by 2.5 km above today's toe
    up = ss > np.nanmax(np.where(np.isfinite(crest), ss, np.nan)) if np.isfinite(crest).any() else ss > toe_s
    fade = np.clip(1 - (ss - toe_s) / C.UP_FADE_M, 0, 1)
    dz = np.where(np.isfinite(crest), crest - zflow, np.nan)
    # with no crests, the east lateral moraine's height above the valley floor near the toe,
    # about 124 m (Bath Geological Society, 1985), less ~20 m of ice still at today's toe
    dz_toe = np.nanmedian(dz[max(0, toe_i - 20):toe_i + 20]) if np.isfinite(dz[max(0, toe_i - 20):toe_i + 20]).any() else C.DZ_TOE_M
    info["checks"]["thickening_at_toe_m"] = round(float(dz_toe), 1)
    dz_fill = np.where(np.isfinite(dz), dz, C.UP_MIN_M + (dz_toe - C.UP_MIN_M) * fade)
    s1844 = zflow + np.clip(dz_fill, 0, 250)
    plastic = bflow + K * np.sqrt(np.maximum(ss, 0))           # snout from the 1844 front
    # the plastic snout shapes the front: across the forefield and just above today's toe,
    # blending out over the next 500 m (up-glacier the moraine-based surface stands)
    wcap = np.clip((toe_s + 800 - ss) / 500, 0, 1)
    s1844 = np.minimum(s1844, plastic) * wcap + s1844 * (1 - wcap)
    s1844 = np.maximum(s1844, zflow)
    s1844 = ndimage.gaussian_filter1d(s1844, 8, mode="nearest")
    s1844 = np.minimum(s1844, bflow + K * np.sqrt(np.maximum(ss, 0)) * (ss < toe_s) + 1e4 * (ss >= toe_s))
    # spread across the valley: every cell near the flowline takes the surface of its
    # nearest flowline point; ice where that is above the ground, joined to the flowline
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([pr, pc]))
    rr, cc = np.mgrid[0:H, 0:W]
    dist, j = tree.query(np.column_stack([rr.ravel(), cc.ravel()]), workers=-1)
    dist = dist.reshape(SHAPE) * C.RES
    j = j.reshape(SHAPE)
    # a rounded snout: distance back from the front, less a parabola across the valley
    # (radius SNOUT_R_M), so the plastic thickness cap draws a lobe, not a straight edge
    along = (ss[j] - dist ** 2 / (2 * C.SNOUT_R_M)).astype(np.float32)
    s_field = s1844[j]
    ground = np.where(ice_now & (gl_lab > 0), bed, dem)
    # cells whose nearest flowline point is the very first one lie beyond the 1844 front
    corridor = (dist < 1000) & (ss[j] < toe_s + C.UP_FADE_M + 1000) & (j > 2)
    s_field = np.where(ss[j] < toe_s, np.minimum(s_field, ground + K * np.sqrt(np.clip(along, 0, None))), s_field)
    ice44 = corridor & (s_field > ground + 1) & (along > 0)
    lab, _ = ndimage.label(ice44)
    keep = set(np.unique(lab[np.round(pr).astype(int).clip(0, H - 1), np.round(pc).astype(int).clip(0, W - 1)])) - {0}
    ice44 = np.isin(lab, list(keep))
    out = np.where(ice44, np.maximum(ndimage.gaussian_filter(np.where(ice44, s_field, ground), 2), ground), np.nan)
    out = np.where(ath_mask & ice_now, np.fmax(out, s_now), out)
    info["flowline"] = {"rc": [[round(float(a), 1), round(float(b), 1)] for a, b in zip(pr[::10], pc[::10])],
                        "s": [round(float(v)) for v in ss[::10]], "toe_s": round(toe_s),
                        "s1844": [round(float(v), 1) for v in s1844[::10]],
                        "s_now": [round(float(v), 1) for v in zflow[::10]]}
    return out.astype(np.float32), along, toe_s


# -------------------------------------------------------------------- projections
def projections(gl):
    import xarray as xr
    html = http.get(OGGM).text
    files = sorted(set(re.findall(r'href="(run_hydro[^"]+\.nc)"', html)))
    ids = {g["id"] for g in gl}
    batches = sorted({(int(i.split(".")[1]) - 1) // 1000 * 1000 for i in ids})
    out = {}
    for ssp in C.SCENARIOS:
        series = {}
        for b0 in batches:
            batch = f"Batch_{b0}_{b0 + 1000}.nc"
            for f in [f for f in files if f"_{ssp}_" in f and f.endswith(batch)]:
                p = Path("/tmp") / f
                p.write_bytes(http.get(OGGM + f).content)
                try:
                    with xr.open_dataset(p) as ds:
                        sel = [i for i in ds["rgi_id"].values if i in ids]
                        if not sel:
                            continue
                        v = ds["volume"].sel(rgi_id=sel).load()
                        yrs = [int(t) for t in ds["time"].values]
                        for i in sel:
                            series.setdefault(i, []).append(v.sel(rgi_id=i).values.astype(float))
                finally:
                    p.unlink()
        rel = {}
        n_t = min(len(r) for runs in series.values() for r in runs) if series else 0
        yrs = yrs[:n_t]
        for i, runs in series.items():
            a = np.array([r[:n_t] for r in runs])
            base = a[:, yrs.index(2020)][:, None]
            r = a / np.where(base > 0, base, np.nan)
            rel[i] = {"median": np.nanmedian(r, 0).round(4).tolist(), "min": np.nanmin(r, 0).round(4).tolist(),
                      "max": np.nanmax(r, 0).round(4).tolist(), "n": int(len(runs))}
        out[ssp] = {"years": yrs, "glaciers": rel}
        a = rel.get(C.ATHABASCA)
        if a:
            say(f"Athabasca {ssp}: {a['n']} models, 2100 volume {100 * a['median'][yrs.index(2100)]:.0f}% of 2020 "
                f"({100 * a['min'][yrs.index(2100)]:.0f}-{100 * a['max'][yrs.index(2100)]:.0f}%)")
    return out


# -------------------------------------------------------------------- main
def main():
    C.DATA.mkdir(parents=True, exist_ok=True)
    dem = terrain()
    gl, lab, ath_id = glaciers()
    ath_mask = lab == ath_id
    tex, ndsi = imagery()
    Image.fromarray(tex).save(C.DATA / "texture.png")
    # today's ice: inside an RGI outline, and either snow/ice to Sentinel-2 or well inside the outline
    inside = ndimage.distance_transform_edt(lab > 0) * C.RES
    ice_now = (lab > 0) & ((ndsi > 0.25) | ((inside > 150) & (ndsi > -0.5)))
    ice_now = ndimage.binary_opening(ice_now, iterations=1)
    lbl, _ = ndimage.label(ice_now)
    sizes = ndimage.sum(ice_now, lbl, range(1, lbl.max() + 1))
    ice_now = np.isin(lbl, 1 + np.flatnonzero(sizes * C.RES ** 2 > 20000))
    th, got = thickness(gl, dem)
    if got == 0:
        th = sia_thickness(dem, lab > 0)
        info["thickness_source"] = "shallow-ice estimate from surface slope (consensus files unavailable)"
    # taper the thickness to nothing at today's ice edge, so the bed doesn't step down in a
    # cliff where the outline ends (the consensus grids end abruptly at their outlines)
    edge = ndimage.distance_transform_edt(ice_now) * C.RES
    th = np.where(ice_now, th, 0) * np.clip(edge / 250.0, 0, 1) ** 0.6
    th = ndimage.gaussian_filter(th, 3) * ice_now
    bed = np.minimum(dem, dem - th)
    vol_ath = float((th * ath_mask).sum() * C.RES ** 2)
    say(f"Athabasca ice today: {vol_ath / 1e9:.2f} km3 in {ath_mask.sum() * C.RES ** 2 / 1e6:.1f} km2 of outline")
    s1844, along, toe_s = reconstruct_1844(dem, bed, ice_now, ath_mask, lab)
    ground = np.where(ice_now, bed, dem)
    v44 = float(np.nansum(np.where(np.isfinite(s1844), s1844 - ground, 0)) * C.RES ** 2)
    v_tongue_now = float(np.nansum(np.where(np.isfinite(s1844) & ice_now, dem - bed, 0)) * C.RES ** 2)
    info["checks"]["tongue_volume_1844_km3"] = round(v44 / 1e9, 3)
    info["checks"]["tongue_volume_now_km3"] = round(v_tongue_now / 1e9, 3)
    say(f"reconstructed tongue (the part that changes): {v44 / 1e9:.3f} km3 in 1844, {v_tongue_now / 1e9:.3f} today "
        f"({100 * (1 - v_tongue_now / max(v44, 1)):.0f}% lost)")
    save("dem.tif", dem)
    save("bed.tif", bed)
    save("glaciers.tif", lab, "int16")
    save("ice_now.tif", ice_now, "uint8")
    save("s1844.tif", np.nan_to_num(s1844, nan=-9999), nodata=-9999)
    save("along.tif", along)
    info["projections"] = projections(gl)
    info.update({"crs": C.CRS, "res": C.RES, "transform": list(T)[:6], "shape": [H, W],
                 "glaciers": [{k: v for k, v in g.items() if k != "geom"} for g in gl],
                 "athabasca_index": ath_id, "volume_now_athabasca_m3": vol_ath, "toe_s": toe_s})
    (C.DATA / "place.json").write_text(json.dumps(info, default=float))
    # checks against what was measured: Reynolds & Young (1997) estimated 0.234 km3 lost
    # from 1919 to 1979 from survey maps; the glacier has lost over half its volume in
    # about 125 years (Parks Canada, via Wikipedia)
    from surfaces import Melt
    m = Melt()
    msk = (m.lab == m.ath) | m.has44
    v = {y: m.volume(y, msk) for y in (1844, 1900, 1919, 1979, C.TODAY)}
    info["checks"]["loss_1919_1979_km3"] = round((v[1919] - v[1979]) / 1e9, 3)
    info["checks"]["volume_km3"] = {y: round(x / 1e9, 3) for y, x in v.items()}
    say(f"check: 1919-79 loss {(v[1919] - v[1979]) / 1e9:.3f} km3 (maps: 0.234); "
        f"since 1900 {100 * (1 - v[C.TODAY] / v[1900]):.0f}% lost (reported: over half)")
    (C.DATA / "place.json").write_text(json.dumps(info, default=float))
    # a quick look: texture with today's ice edge, the 1844 ice edge and the flowline
    q = Image.fromarray(tex).convert("RGB")
    from PIL import ImageDraw
    d = ImageDraw.Draw(q)
    e44 = np.isfinite(s1844) & ~ndimage.binary_erosion(np.isfinite(s1844))
    en = ice_now & ~ndimage.binary_erosion(ice_now)
    a = np.array(q)
    a[e44] = (255, 80, 40)
    a[en] = (40, 220, 255)
    q = Image.fromarray(a)
    d = ImageDraw.Draw(q)
    fl = info["flowline"]["rc"]
    d.line([(c, r) for r, c in fl], fill=(255, 230, 0), width=2)
    r0, c0 = rc(*xy(*C.ICEFIELD_CENTRE))
    d.ellipse([c0 - 6, r0 - 6, c0 + 6, r0 + 6], outline=(255, 255, 255), width=2)
    q.save(C.DATA / "quicklook.png")


if __name__ == "__main__":
    main()
