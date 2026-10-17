"""What data is there for the Athabasca Glacier melt film? (runs on GitHub Actions)

1. Canada's lidar terrain (HRDEM, 1-2 m) over the glacier and the lookout: the STAC catalog.
2. The glacier outline and its RGI 6 ID (the inventory OGGM's projections use).
3. OGGM's per-glacier projections to 2100: which files hold the Athabasca, and its
   volume, area and length under a few scenarios.

Findings are printed as lines starting "= " and written to melt/probe.json.

    python melt/probe.py
"""
from __future__ import annotations

import io
import json
import re
import sys
import zipfile
from pathlib import Path

import httpx

OUT = Path(__file__).resolve().parent / "probe.json"
BBOX = (-117.32, 52.15, -117.15, 52.25)          # the glacier, the toe and Icefield Centre
TONGUE = (-117.245, 52.200)                       # a point on the Athabasca's tongue
ICEFIELD_CENTRE = (-117.2244, 52.2197)            # GeoVistas brochure: N52 13.179, W117 13.465
OGGM = "https://cluster.klima.uni-bremen.de/~oggm/oggm-standard-projections/oggm_v16/2025.6/w5e5/per_glacier_spinup/CMIP6/2100/RGI02/"
res: dict = {}
http = httpx.Client(timeout=180, follow_redirects=True)


def say(s):
    print("= " + s, flush=True)


def hrdem():
    try:
        cols = http.get("https://datacube.services.geo.ca/stac/api/collections").json()["collections"]
        ids = [c["id"] for c in cols if "hrdem" in c["id"].lower() or "mrdem" in c["id"].lower()]
        say(f"elevation collections: {ids}")
        found = {}
        for cid in ids:
            r = http.get("https://datacube.services.geo.ca/stac/api/search",
                         params={"collections": cid, "bbox": ",".join(map(str, BBOX)), "limit": 50}).json()
            feats = r.get("features", [])
            found[cid] = [{"id": f["id"], "datetime": f["properties"].get("datetime") or f["properties"].get("start_datetime"),
                           "assets": {k: v.get("href") for k, v in f["assets"].items()}} for f in feats]
            say(f"{cid}: {len(feats)} items over the glacier" +
                (f" e.g. {feats[0]['id']} ({found[cid][0]['datetime']}) assets {list(feats[0]['assets'])}" if feats else ""))
        res["hrdem"] = found
        import numpy as np
        import rasterio
        from rasterio.warp import transform_bounds
        from rasterio.windows import from_bounds
        res["hrdem_check"] = {}
        for cid in ("hrdem-mosaic-1m", "hrdem-mosaic-2m", "mrdem-30"):
            for it in found.get(cid, []):
                url = it["assets"].get("dtm")
                if not url:
                    continue
                with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"), rasterio.open("/vsicurl/" + url) as src:
                    l, b_, r_, t = transform_bounds("EPSG:4326", src.crs, *BBOX)
                    win = from_bounds(l, b_, r_, t, src.transform)
                    step = max(1, int(round(10 / src.res[0])))          # sample on a ~10 m grid
                    ov = src.read(1, window=win, out_shape=(int(win.height // step), int(win.width // step)),
                                  masked=True)
                    valid = float((~ov.mask).mean()) if np.ma.isMaskedArray(ov) else 1.0
                    info = {"crs": str(src.crs), "res": src.res, "valid_fraction": round(valid, 3),
                            "zmin": float(ov.min()) if valid else None, "zmax": float(ov.max()) if valid else None}
                    # the Athabasca tongue and Icefield Centre specifically
                    for nm, (lon, lat) in (("tongue", TONGUE), ("centre", ICEFIELD_CENTRE)):
                        from rasterio.warp import transform as tf
                        x, y = tf("EPSG:4326", src.crs, [lon], [lat])
                        v = list(src.sample([(x[0], y[0])], masked=True))[0]
                        info[nm] = None if np.ma.is_masked(v[0]) else float(v[0])
                    res["hrdem_check"][cid] = info
                    say(f"{cid} DTM over the glacier: {src.res[0]:g} m, {100 * valid:.0f}% has data, "
                        f"tongue {info['tongue']}, Icefield Centre {info['centre']}")
    except Exception as e:
        say(f"HRDEM STAC failed: {e!r}")


def rgi():
    from shapely.geometry import Point, shape
    import shapefile  # pyshp
    urls = []
    for base in ("https://cluster.klima.uni-bremen.de/~oggm/rgi/", "https://cluster.klima.uni-bremen.de/~oggm/rgi/RGI62/"):
        try:
            html = http.get(base).text
            links = re.findall(r'href="([^"?]+)"', html)
            say(f"listing {base}: {[x for x in links if not x.startswith('/')][:40]}")
            urls += [base + x for x in links if re.search(r"(^|_)02_rgi6\d.*\.zip$", x)]
        except Exception as e:
            say(f"listing {base} failed: {e!r}")
    urls += ["https://cluster.klima.uni-bremen.de/~oggm/rgi/RGIV62/02_rgi62_WesternCanadaUS.zip",
             "https://cluster.klima.uni-bremen.de/~oggm/rgi/rgi62.zip"]
    for url in urls:
        try:
            z = zipfile.ZipFile(io.BytesIO(http.get(url).content))
            if not any(n.endswith(".shp") for n in z.namelist()):     # a zip of region zips
                inner = [n for n in z.namelist() if re.search(r"(^|/)02_rgi6.*\.zip$", n)][0]
                z = zipfile.ZipFile(io.BytesIO(z.read(inner)))
            shp = [n for n in z.namelist() if n.endswith(".shp")][0]
            base = shp[:-4]
            r = shapefile.Reader(shp=io.BytesIO(z.read(base + ".shp")), dbf=io.BytesIO(z.read(base + ".dbf")),
                                 shx=io.BytesIO(z.read(base + ".shx")))
            fields = [f[0] for f in r.fields[1:]]
            near = []
            tp, cp = Point(*TONGUE), Point(*ICEFIELD_CENTRE)
            for sr in r.iterShapeRecords():
                g = shape(sr.shape.__geo_interface__)
                if g.distance(cp) < 0.06:
                    rec = dict(zip(fields, sr.record))
                    near.append({"id": rec["RGIId"], "name": rec.get("Name"), "area_km2": rec["Area"],
                                 "zmin": rec["Zmin"], "zmax": rec["Zmax"], "lmax_m": rec.get("Lmax"),
                                 "contains_tongue_point": g.contains(tp), "dist_to_centre_km": round(g.distance(cp) * 70, 2),
                                 "geom": g.__geo_interface__})
            near.sort(key=lambda x: -x["area_km2"])
            for n in near[:8]:
                say(f"RGI {n['id']} {n['name'] or '-'}: {n['area_km2']} km2, {n['zmin']}-{n['zmax']} m, "
                    f"tongue point inside: {n['contains_tongue_point']}")
            res["rgi"] = near
            say(f"RGI from {url}")
            return
        except Exception as e:
            say(f"RGI from {url} failed: {e!r}")


def oggm():
    import xarray as xr
    try:
        html = http.get(OGGM).text
        files = sorted(set(re.findall(r'href="(run_hydro[^"]+\.nc)"', html)))
        say(f"OGGM RGI02 CMIP6/2100: {len(files)} files, e.g. {files[:3]}")
        files = [f for f in files if re.search(r"_ssp\d+_", f)]
        gcms = sorted({re.search(r"_([A-Za-z0-9.\-]+)_ssp\d+_", f).group(1) for f in files if "_ssp" in f})
        ssps = sorted({re.search(r"_(ssp\d+)_", f).group(1) for f in files if "_ssp" in f})
        say(f"GCMs: {len(gcms)} {gcms[:20]}; scenarios: {ssps}")
        res["oggm"] = {"base": OGGM, "gcms": gcms, "ssps": ssps, "n_files": len(files)}
        ath = next((n["id"] for n in res.get("rgi", []) if n["contains_tongue_point"]), None)
        if not ath:
            say("no Athabasca RGI id; skipping projections")
            return
        num = int(ath.split(".")[1])
        b0 = (num - 1) // 1000 * 1000
        batch = f"Batch_{b0}_{b0 + 1000}.nc"
        say(f"Athabasca is {ath}: looking in *{batch}")
        res["oggm"]["athabasca"] = {"rgi_id": ath, "batch": batch, "series": {}}
        for ssp in ("ssp126", "ssp245", "ssp370", "ssp585"):
            fs = [f for f in files if f"_{ssp}_" in f and f.endswith(batch)]
            vols = []
            for f in fs:
                p = Path("/tmp") / f
                p.write_bytes(http.get(OGGM + f).content)
                with xr.open_dataset(p) as ds:
                    if ath not in ds["rgi_id"].values:
                        continue
                    s = ds.sel(rgi_id=ath)
                    vols.append({"gcm": re.search(r"_([A-Za-z0-9.\-]+)_ssp\d+_", f).group(1),
                                 "time": [int(t) for t in s["time"].values],
                                 "volume": [float(v) for v in s["volume"].values],
                                 "area": [float(v) for v in s["area"].values],
                                 "length": [float(v) for v in s["length"].values]})
                p.unlink()
            res["oggm"]["athabasca"]["series"][ssp] = vols
            if vols:
                import numpy as np
                t = vols[0]["time"]
                v = np.array([x["volume"] for x in vols])
                i20, i50, i00 = t.index(2020), t.index(2050), t.index(2100)
                med = np.nanmedian(v, axis=0)
                say(f"Athabasca {ssp}: {len(vols)} GCMs, median volume 2020 {med[i20] / 1e9:.2f} km3, "
                    f"2050 {100 * med[i50] / med[i20]:.0f}%, 2100 {100 * med[i00] / med[i20]:.0f}% of 2020 "
                    f"(range {100 * np.nanmin(v[:, i00]) / med[i20]:.0f}-{100 * np.nanmax(v[:, i00]) / med[i20]:.0f}%)")
    except Exception as e:
        say(f"OGGM failed: {e!r}")


hrdem()
rgi()
oggm()
OUT.write_text(json.dumps(res))
say(f"wrote {OUT.name}")
