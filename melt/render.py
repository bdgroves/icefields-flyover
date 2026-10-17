"""The Athabasca Glacier from the Icefield Centre lookout, 1844 to 2100, with forge3d.

The camera doesn't move; time does. Every frame loads that year's ice surface as the
terrain and paints that year's ice onto the Sentinel-2 photo.

    python melt/render.py --stills "1844,1919,1979,2025,2050,2100" --size 1280x720 --lite
    python melt/render.py --frames-dir out/frames --chunk 0 --chunks 40 --lite
    python melt/render.py --card              # 2100 under each scenario, side by side
    python melt/render.py --encode out/frames # -> out/athabasca_melt.mp4 (+ the card)
"""
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from surfaces import Melt  # noqa: E402

import forge3d as f3d  # noqa: E402

FPS = 30
# (start s, end s, start year, end year)
TIMELINE = [(0, 4, 1844, 1844), (4, 34, 1844, C.TODAY), (34, 38, C.TODAY, C.TODAY),
            (38, 63, C.TODAY, 2100), (63, 68, 2100, 2100)]
CARD_S = 7
FOV = 52.0
ZENITH = np.array([52, 112, 186]) / 255.0
HORIZON = np.array([200, 218, 232]) / 255.0
PBR = {"enabled": True, "exposure": 0.95, "msaa": 4, "shadow_technique": "none",
       "height_ao": {"enabled": True, "strength": 0.6, "max_distance": 200.0},
       "sun_visibility": {"enabled": True, "mode": "soft", "max_distance": 6000.0}}
TERRAIN = {"zscale": 1.0, "ambient": 0.42}
SUN_INTENSITY = 0.75
WHITE, HALO = (253, 250, 244), (24, 28, 34)
ICE_RGB = None      # sampled from today's glacier in the photo
TILL_RGB = None     # sampled from the forefield the glacier left


def year_at(t: float) -> float:
    for a, b, y0, y1 in TIMELINE:
        if t <= b:
            f = (t - a) / (b - a) if b > a else 0
            return y0 + (y1 - y0) * f
    return TIMELINE[-1][3]


N_FRAMES = int(TIMELINE[-1][1] * FPS)


# ---------------------------------------------------------------- viewer plumbing
class _Tee:
    def __init__(self, stream, log):
        self._s, self._log = stream, log

    def _keep(self, line):
        if line:
            self._log.write(line)
            self._log.flush()
        return line

    def readline(self, *a):
        return self._keep(self._s.readline(*a))

    def read(self, *a):
        return self._keep(self._s.read(*a))

    def __iter__(self):
        for line in self._s:
            yield self._keep(line)

    def __getattr__(self, n):
        return getattr(self._s, n)


def open_viewer(log_path, **kw):
    import forge3d.viewer as fv
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    real = fv.subprocess.Popen

    def popen(*a, **k):
        p = real(*a, **k)
        if p.stdout is not None:
            p.stdout = _Tee(p.stdout, log)
        return p

    fv.subprocess.Popen = popen
    try:
        return f3d.open_viewer_async(timeout=180.0, **kw)
    finally:
        fv.subprocess.Popen = real


def camera(eye, aim, min_h, fov=FOV):
    def world(p):
        return np.array([p[0], p[2] - min_h, -p[1]])
    e, a = world(eye), world(aim)
    off = e - a
    r = float(np.linalg.norm(off))
    if r > 600:
        a = e - off * (600 / r)
        off = e - a
        r = 600.0
    return {"cmd": "set_terrain_camera", "phi_deg": math.degrees(math.atan2(off[2], off[0])),
            "theta_deg": math.degrees(math.acos(np.clip(off[1] / r, -1, 1))), "radius": r,
            "fov_deg": fov, "target": [float(v) for v in a]}


def project(pts, eye, aim, fov, w, h):
    pts = np.atleast_2d(pts).astype(float)
    fwd = (aim - eye) / np.linalg.norm(aim - eye)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    d = pts - eye
    z = d @ fwd
    t = math.tan(math.radians(fov) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (d @ right) / z / (t * w / h)
        y = (d @ up) / z / t
    px, py = (x + 1) * w / 2, (1 - y) * h / 2
    px[z <= 0] = py[z <= 0] = np.nan
    return px, py, np.linalg.norm(d, axis=1)


def sky_mask(img):
    lo, hi = img.min(axis=2), img.max(axis=2)
    lum = img.mean(axis=2).astype(np.float32)
    smooth = np.abs(lum - ndimage.uniform_filter(lum, 3)) < 1.5
    cand = (lo >= 228) & (hi - lo <= 10) & smooth
    lab, _ = ndimage.label(cand)
    sky = np.isin(lab, np.unique(lab[0][lab[0] > 0]))
    return ndimage.binary_dilation(ndimage.binary_closing(sky, iterations=2) | sky, iterations=1)


def finish(png):
    img = np.asarray(Image.open(png).convert("RGB")).astype(np.int16)
    sky = sky_mask(img)
    h = img.shape[0]
    first = np.argmin(sky, axis=0)
    first[sky.all(axis=0)] = h
    horizon = int(np.clip(np.median(first), 0.25 * h, h))
    x = img.astype(np.float64) / 255.0
    lum = x @ np.array([0.2126, 0.7152, 0.0722])
    hi = np.clip((lum - 0.25) / 0.6, 0, 1)[..., None]
    x = x * (1 + hi * np.array([0.05, 0.02, -0.03]))
    x = np.clip((x - 0.04) / 0.95, 0, 1)
    x = np.clip((x - 0.5) * 1.12 + 0.5, 0, 1)
    g = (x @ np.array([0.2126, 0.7152, 0.0722]))[..., None]
    x = np.clip(g + (x - g) * 1.15, 0, 1)
    below = np.arange(h)[:, None] - first[None, :]
    f = 0.18 * (1 - np.clip(below / (0.22 * h), 0, 1)[..., None]) ** 2
    x = x * (1 - f) + HORIZON * f
    t = np.clip(np.arange(h) / max(horizon, 1), 0, 1)[:, None] ** 1.4
    paint = ZENITH * (1 - t) + HORIZON * t
    a = ndimage.gaussian_filter(sky.astype(float), 0.8)[..., None]
    x = x * (1 - a) + np.broadcast_to(paint[:, None, :], x.shape) * a
    return np.clip(x * 255 + 0.5, 0, 255).astype(np.uint8)


def font(size, bold=True):
    names = (("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "arialbd.ttf")
             if bold else ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "arial.ttf"))
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


# ---------------------------------------------------------------- the scene
class Scene:
    def __init__(self, scenario=C.SCENARIO):
        self.m = Melt(scenario)
        info = self.m.info
        with rasterio.open(C.DATA / "dem.tif") as s:
            self.T = s.transform
            self.prof = s.profile
        self.tex = np.asarray(Image.open(C.DATA / "texture.png").convert("RGB")).astype(np.float32)
        lab, ice_now = self.m.lab, self.m.ice_now
        # colours from the photo itself: today's lower glacier, and the bare forefield
        low = ice_now & (lab == self.m.ath) & (self.m.dem < np.percentile(self.m.dem[ice_now & (lab == self.m.ath)], 30))
        fore = self.m.has44 & ~ice_now
        self.ice_rgb = np.median(self.tex[low], axis=0) if low.any() else np.array([205, 210, 214.0])
        self.till_rgb = np.median(self.tex[fore], axis=0) if fore.any() else np.array([128, 120, 108.0])
        rng = np.random.default_rng(7)
        self.noise = ndimage.gaussian_filter(rng.standard_normal(self.m.dem.shape), 1.2)
        self.noise /= self.noise.std() + 1e-9
        # camera: 250 m down-valley from Icefield Centre, 60 m up, looking up the tongue
        fl = info["flowline"]
        rc = np.array(fl["rc"]); s_ = np.array(fl["s"])
        def to_xy(r, c):
            x, y = self.T * (c + 0.5, r + 0.5)
            return np.array([x, y])
        p0, p1 = to_xy(*rc[0]), to_xy(*rc[min(len(rc) - 1, 3)])
        back = (p0 - p1) / np.linalg.norm(p0 - p1)
        e = p0 + back * C.EYE_BACK_M
        er, ec = ~self.T * (e[0], e[1])
        ground = float(ndimage.map_coordinates(self.m.dem, [[er], [ec]], order=1)[0])
        r0, c0 = rc[0]
        z0 = float(ndimage.map_coordinates(self.m.dem, [[r0], [c0]], order=1)[0])
        self.eye = np.array([e[0], e[1], max(ground + 25, z0 + C.EYE_UP_M)])
        k = int(np.argmin(np.abs(s_ - (self.m.toe_s + C.AIM_UP_M))))
        a = to_xy(*rc[k])
        self.aim = np.array([a[0], a[1], fl["s_now"][k] - C.AIM_DOWN_M])
        self.flow = (rc, s_)
        # labels
        from rasterio.warp import transform
        self.labels = []
        for _, (lat, lon, text) in C.LABELS.items():
            x, y = transform("EPSG:4326", C.CRS, [lon], [lat])
            r, c = ~self.T * (x[0], y[0])
            if 0 <= r < self.m.dem.shape[0] and 0 <= c < self.m.dem.shape[1]:
                z = float(ndimage.map_coordinates(self.m.dem, [[r], [c]], order=1)[0])
                self.labels.append((np.array([x[0], y[0], z]), text))
        self.chart = self._chart_data()

    def toe_point(self, year):
        rc, s_ = self.flow
        s = self.m.retreat(year)
        r = np.interp(s, s_, rc[:, 0]); c = np.interp(s, s_, rc[:, 1])
        x, y = self.T * (c + 0.5, r + 0.5)
        z = float(ndimage.map_coordinates(self.m.ground, [[r], [c]], order=1)[0])
        return np.array([x, y, z])

    def _chart_data(self):
        mask = (self.m.lab == self.m.ath) | self.m.has44
        yrs = list(range(1844, C.TODAY + 1, 4)) + [C.TODAY] + list(range(C.TODAY + 1, 2101))
        v = np.array([self.m.volume(y, mask) for y in yrs])
        v0 = v[0]
        pr = self.m.info["projections"][self.m.scenario]
        gid = self.m.info["glaciers"][self.m.ath - 1]["id"]
        g = pr["glaciers"].get(gid)
        band = None
        if g:
            py = np.array(pr["years"], float)
            vt = self.m.volume(C.TODAY, mask)
            f = lambda arr: np.interp(np.arange(C.TODAY, 2101), py, arr) / np.interp(C.TODAY, py, arr)
            # the model spread applies to the whole glacier; scale today's tongue+glacier volume by it
            band = (np.arange(C.TODAY, 2101), vt * f(g["min"]) / v0, vt * f(g["max"]) / v0)
        return {"years": np.array(yrs), "frac": v / v0, "band": band, "n": g["n"] if g else 0}

    def texture(self, ice):
        soft = ndimage.gaussian_filter(ice.astype(np.float32), 1.0)
        now = ndimage.gaussian_filter(self.m.ice_now.astype(np.float32), 1.0)
        out = self.tex.copy()
        add = np.clip(soft - now, 0, 1)[..., None]           # ice that's gone today
        gone = np.clip(now - soft, 0, 1)[..., None]          # ice that isn't there yet
        ice_px = self.ice_rgb * (1 + 0.05 * self.noise[..., None])
        edge = np.clip(1 - ndimage.distance_transform_edt(ice) * C.RES / 120, 0, 1)[..., None]
        ice_px = ice_px * (1 - 0.25 * edge) + self.till_rgb * 0.25 * edge     # dirtier margins
        till_px = self.till_rgb * (1 + 0.08 * self.noise[..., None])
        # water the photo shows where ice stood (lakes the glacier has since uncovered)
        t = self.tex
        water = ((t[..., 1] > t[..., 0] + 18) & (t[..., 2] > t[..., 0] + 8))[..., None] & ice[..., None]
        add = np.maximum(add, water * soft[..., None])
        out = out * (1 - add) + ice_px * add
        out = out * (1 - gone) + till_px * gone
        return np.clip(out, 0, 255).astype(np.uint8)

    def overlays(self, rgb, year, w, h):
        img = Image.fromarray(rgb)
        d = ImageDraw.Draw(img, "RGBA")
        S = h / 1080
        f_year, f_big, f_mid, f_small = font(int(96 * S)), font(int(44 * S)), font(int(28 * S)), font(int(19 * S), False)
        st = max(2, int(3 * S))
        # peak names
        hud = [(0, 0, 900 * S, 200 * S), (w - 500 * S, h - 360 * S, w, h)]      # year/phase, chart
        for p, text in self.labels:
            px, py, _ = project(p, self.eye, self.aim, FOV, w, h)
            if not np.isfinite(px[0]):
                continue
            tw = d.textlength(text, font=f_mid)
            x0, x1, ty = px[0] - tw / 2, px[0] + tw / 2, py[0] - 30 * S
            inside = 20 * S < x0 and x1 < w - 20 * S and 60 * S < ty and py[0] < h
            clash = any(x1 > a and x0 < c and ty > b - 40 * S and ty - 40 * S < dd for a, b, c, dd in hud)
            if inside and not clash:
                d.line([(px[0], py[0] - 4), (px[0], py[0] - 26 * S)], fill=WHITE + (220,), width=max(1, st // 2))
                d.text((px[0], py[0] - 30 * S), text, font=f_mid, fill=WHITE, anchor="md", stroke_width=st, stroke_fill=HALO)
        # where the front stood: posts at 1844, 1900, 1950, 2000 once passed
        if year > 1844:
            for yy in (1844, 1900, 1950, 2000):
                if year < yy:
                    break
                a = float(np.clip((year - yy) / 6, 0, 1))
                p = self.toe_point(yy)
                px, py, _ = project(p, self.eye, self.aim, FOV, w, h)
                if np.isfinite(px[0]) and 0 < px[0] < w and 0 < py[0] < h:
                    col = (255, 214, 120, int(235 * a))
                    d.line([(px[0], py[0]), (px[0], py[0] - 40 * S)], fill=col, width=st)
                    d.text((px[0], py[0] - 44 * S), f"front, {yy if yy == 1844 else '~' + str(yy)}", font=f_small,
                           fill=col, anchor="md", stroke_width=st, stroke_fill=HALO + (int(220 * a),))
        # year and what it is
        yi = int(round(year))
        d.text((60 * S, 50 * S), str(yi), font=f_year, fill=WHITE, stroke_width=st + 1, stroke_fill=HALO)
        if year <= 1844.01:
            phase = "Little Ice Age maximum · reconstruction"
        elif year < C.TODAY - 0.01:
            phase = "Reconstruction from moraines and the glacier's dated front"
        elif year <= C.TODAY + 0.01:
            phase = "Today"
        else:
            n = self.chart["n"]
            phase = f"Projection · {C.SCENARIO_NAMES[self.m.scenario]} · median of {n} climate models (OGGM)"
        col = (255, 196, 120) if year > C.TODAY + 0.01 else WHITE
        d.text((64 * S, 160 * S), phase, font=f_mid, fill=col, stroke_width=st, stroke_fill=HALO)
        self._draw_chart(d, year, w, h, S)
        d.text((60 * S, h - 34 * S),
               "Terrain NRCan MRDEM 30 m · imagery Sentinel-2 · ice thickness Farinotti et al. 2019 · projection OGGM v1.6.3",
               font=f_small, fill=WHITE + (210,), stroke_width=max(1, st - 1), stroke_fill=HALO)
        return np.asarray(img)

    def _draw_chart(self, d, year, w, h, S):
        x0, y0, cw, ch = w - 470 * S, h - 300 * S, 410 * S, 220 * S
        d.rounded_rectangle([x0 - 18 * S, y0 - 52 * S, x0 + cw + 18 * S, y0 + ch + 40 * S], 10 * S, fill=(20, 24, 30, 150))
        fs, fm = font(int(17 * S), False), font(int(20 * S))
        d.text((x0, y0 - 40 * S), "Ice in the Athabasca, % of 1844", font=fm, fill=WHITE)
        X = lambda yy: x0 + (yy - 1844) / (2100 - 1844) * cw
        Y = lambda f: y0 + ch * (1 - f)
        for f in (0, 0.5, 1):
            d.line([(x0, Y(f)), (x0 + cw, Y(f))], fill=(255, 255, 255, 50), width=1)
            d.text((x0 - 6 * S, Y(f)), f"{int(f * 100)}", font=fs, fill=(230, 230, 230), anchor="rm")
        for yy in (1844, 1900, 1950, 2000, 2050, 2100):
            d.text((X(yy), y0 + ch + 8 * S), str(yy), font=fs, fill=(230, 230, 230), anchor="mt")
        c = self.chart
        if c["band"] is not None:
            by, lo, hi = c["band"]
            poly = [(X(a), Y(b)) for a, b in zip(by, hi)] + [(X(a), Y(b)) for a, b in zip(by[::-1], lo[::-1])]
            d.polygon(poly, fill=(255, 196, 120, 55))
        yrs, fr = c["years"], c["frac"]
        past = yrs <= C.TODAY
        d.line([(X(a), Y(b)) for a, b in zip(yrs[past], fr[past])], fill=WHITE + (130,), width=max(2, int(2 * S)))
        d.line([(X(a), Y(b)) for a, b in zip(yrs[~past], fr[~past])], fill=(255, 196, 120, 130), width=max(2, int(2 * S)))
        upto = yrs <= year
        if upto.sum() > 1:
            d.line([(X(a), Y(b)) for a, b in zip(yrs[upto], fr[upto])], fill=WHITE, width=max(3, int(4 * S)))
        fy = float(np.interp(year, yrs, fr))
        d.ellipse([X(year) - 7 * S, Y(fy) - 7 * S, X(year) + 7 * S, Y(fy) + 7 * S], fill=(255, 214, 120))
        d.text((X(year) + 10 * S, Y(fy) - 12 * S), f"{fy * 100:.0f}%", font=fm, fill=WHITE, stroke_width=2, stroke_fill=HALO)

    def title(self, rgb, t, w, h):
        a = float(np.clip(min(t / 0.8, (4.2 - t) / 1.0), 0, 1))
        if a <= 0.02:
            return rgb
        img = Image.fromarray(rgb).convert("RGBA")
        o = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(o)
        S = h / 1080
        d.text((w / 2, h * 0.42), "THE ATHABASCA GLACIER", font=font(int(84 * S)), fill=WHITE + (int(255 * a),),
               anchor="mm", stroke_width=4, stroke_fill=HALO + (int(200 * a),))
        d.text((w / 2, h * 0.42 + 80 * S), "1844 to 2100, from the Icefield Centre lookout on the Icefields Parkway",
               font=font(int(32 * S), False), fill=WHITE + (int(255 * a),), anchor="mm", stroke_width=3,
               stroke_fill=HALO + (int(200 * a),))
        return np.asarray(Image.alpha_composite(img, o).convert("RGB"))


# ---------------------------------------------------------------- rendering
def write_terrain(s, path, prof):
    p = dict(prof, dtype="float32", count=1, nodata=None, compress="deflate", predictor=3)
    with rasterio.open(path, "w", **p) as d:
        d.write(s.astype(np.float32), 1)


def render(times, w, h, out_for, lite, scenario=C.SCENARIO, hud=True):
    sc = Scene(scenario)
    C.OUT.mkdir(parents=True, exist_ok=True)
    work = C.OUT / "_work"
    work.mkdir(exist_ok=True)
    s0, _ = sc.m.state(times[0][1])
    tpath = work / "terrain_0.tif"
    write_terrain(s0, tpath, sc.prof)
    sun = f3d.sun_position_utc(*C.SUN_LATLON, *C.SUN_UTC)
    print(f"Sun: azimuth {sun.azimuth:.0f}, {sun.elevation:.0f} up; eye {sc.eye.round(0)}, aim {sc.aim.round(0)}")
    pbr = dict(PBR, msaa=4 if lite else 8)
    viewer = open_viewer(C.OUT / "viewer.log", width=w, height=h, terrain_path=str(tpath), fov_deg=FOV,
                         title="Athabasca Glacier")
    snap = work / "_frame.png"
    start = time.perf_counter()
    try:
        for n, (t, year) in enumerate(times):
            s, ice = sc.m.state(year)
            tp = work / f"terrain_{n % 2}.tif"
            write_terrain(s, tp, sc.prof)
            texp = work / f"tex_{n % 2}.png"
            Image.fromarray(sc.texture(ice)).save(texp)
            if n > 0:
                viewer.load_terrain(str(tp))
            if n > 0:      # one photo layer at a time: drop the previous frame's
                try:
                    viewer.send_ipc({"cmd": "remove_overlay", "id": n - 1})
                except Exception as e:
                    print(f"  remove_overlay: {e!r}")
            viewer.load_overlay("photo", texp, extent=(0.0, 0.0, 1.0, 1.0), z_order=0, preserve_colors=False)
            viewer.send_ipc({"cmd": "set_terrain_sun", "azimuth_deg": float(sun.azimuth),
                             "elevation_deg": float(sun.elevation), "intensity": SUN_INTENSITY})
            viewer.send_ipc({"cmd": "set_terrain_pbr", **pbr})
            viewer.send_ipc({"cmd": "set_terrain", **TERRAIN})
            viewer.send_ipc(camera(sc.eye, sc.aim, float(s.min()), FOV))
            if n == 0:
                time.sleep(2.0)
            viewer.snapshot(snap, w, h)
            rgb = finish(snap)
            if hud:
                rgb = sc.overlays(rgb, year, w, h)
                rgb = sc.title(rgb, t, w, h)
            out_for(n, t, year, rgb)
            el = time.perf_counter() - start
            print(f"  {n + 1}/{len(times)}  year {year:.1f}  ~{el / (n + 1) * (len(times) - n - 1) / 60:.0f} min left", flush=True)
    finally:
        viewer.close()


def card(w, h, lite):
    """2100 under each scenario, side by side, over the 2100 frame of the film's own scenario."""
    tiles = []
    for ssp in C.SCENARIOS:
        got = {}
        render([(TIMELINE[-1][1], 2100.0)], w // 2, h // 2, lambda n, t, y, rgb: got.update(rgb=rgb), lite,
               scenario=ssp, hud=False)
        tiles.append((ssp, got["rgb"]))
    S = h / 1080
    canvas = Image.new("RGB", (w, h), (18, 22, 28))
    d = ImageDraw.Draw(canvas)
    d.text((w / 2, 70 * S), "The Athabasca in 2100 depends on how much warming we allow", font=font(int(44 * S)),
           fill=WHITE, anchor="mm")
    tw = int(w * 0.31)
    th = int(tw * 9 / 16)
    m0 = Melt(C.SCENARIO)
    for i, (ssp, rgb) in enumerate(tiles):
        x = int(w * (0.025 + i * 0.325))
        y = int(h * 0.2)
        canvas.paste(Image.fromarray(rgb).resize((tw, th)), (x, y))
        m0.set_scenario(ssp)
        pr = m0.info["projections"][ssp]
        gid = m0.info["glaciers"][m0.ath - 1]["id"]
        g = pr["glaciers"].get(gid)
        yrs = pr["years"]
        if g:
            i20, i00 = yrs.index(2020), yrs.index(2100)
            txt = f"{100 * g['median'][i00]:.0f}% of 2020's ice left (models: {100 * g['min'][i00]:.0f}-{100 * g['max'][i00]:.0f}%)"
        else:
            txt = ""
        d.text((x + tw / 2, y + th + 40 * S), C.SCENARIO_NAMES[ssp], font=font(int(30 * S)), fill=WHITE, anchor="mm")
        d.text((x + tw / 2, y + th + 82 * S), txt, font=font(int(24 * S), False), fill=(230, 230, 230), anchor="mm")
    d.text((w / 2, h - 90 * S), "OGGM v1.6.3 per-glacier projections, CMIP6 climate models (Schuster et al.); median and range across models",
           font=font(int(22 * S), False), fill=(200, 200, 200), anchor="mm")
    C.OUT.mkdir(parents=True, exist_ok=True)
    canvas.save(C.OUT / "card.png")
    print("wrote card.png")


def encode(frames: Path, out: Path):
    ff = shutil.which("ffmpeg") or "ffmpeg"
    nums = sorted(int(p.stem.split("_")[1]) for p in frames.glob("frame_*.jpg"))
    if nums != list(range(len(nums))) or len(nums) != N_FRAMES:
        missing = sorted(set(range(N_FRAMES)) - set(nums))
        print(f"{len(nums)} of {N_FRAMES} frames; missing {missing[:20]}")
        return 1
    card_png = C.OUT / "card.png"
    if card_png.exists():                                    # hold the card at the end
        im = Image.open(card_png).convert("RGB").resize(Image.open(frames / "frame_00000.jpg").size)
        for k in range(CARD_S * FPS):
            im.save(frames / f"frame_{N_FRAMES + k:05d}.jpg", quality=95)
    cmd = [ff, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(frames / "frame_%05d.jpg"),
           "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    if subprocess.run(cmd).returncode:
        return 1
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", type=str)
    ap.add_argument("--scenario", default=C.SCENARIO)
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--lite", action="store_true")
    ap.add_argument("--frames-dir", type=str)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--chunks", type=int, default=1)
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--encode", type=str)
    ap.add_argument("--out", type=str)
    a = ap.parse_args()
    w, h = map(int, a.size.lower().split("x"))
    if a.encode:
        return encode(Path(a.encode), Path(a.out) if a.out else C.OUT / "athabasca_melt.mp4")
    if a.card:
        card(w, h, a.lite)
        return 0
    if a.stills:
        years = [float(v) for v in a.stills.split(",")]
        # a still is shown as it appears in the film: find that year's time
        times = []
        for y in years:
            t = next((tt / FPS for tt in range(N_FRAMES) if year_at(tt / FPS) >= y - 1e-6), TIMELINE[-1][1])
            times.append((max(t, 4.5) if y > 1844 else 5.0, y))
        d = C.OUT / "stills"
        d.mkdir(parents=True, exist_ok=True)
        render(times, w, h, lambda n, t, y, rgb: Image.fromarray(rgb).save(d / f"still_{int(y)}_{a.scenario}.png"),
               a.lite, a.scenario)
        return 0
    fd = Path(a.frames_dir)
    fd.mkdir(parents=True, exist_ok=True)
    lo, hi = N_FRAMES * a.chunk // a.chunks, N_FRAMES * (a.chunk + 1) // a.chunks
    todo = [i for i in range(lo, hi) if not (fd / f"frame_{i:05d}.jpg").exists()]
    print(f"chunk {a.chunk + 1}/{a.chunks}: frames {lo}-{hi - 1}, {len(todo)} to render")
    if not todo:
        return 0
    times = [(i / FPS, year_at(i / FPS)) for i in todo]

    def save(n, t, y, rgb):
        i = todo[n]
        tmp = fd / f"frame_{i:05d}.part.jpg"
        Image.fromarray(rgb).save(tmp, quality=95)
        tmp.replace(fd / f"frame_{i:05d}.jpg")
    render(times, w, h, save, a.lite, a.scenario)
    return 0


if __name__ == "__main__":
    sys.exit(main())
