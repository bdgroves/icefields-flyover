"""The glacier surface in any year from 1844 to 2100.

Past (1844 to today), the Athabasca only:
  The toe's distance from its 1844 front follows config.RETREAT, scaled so it reaches
  today's toe. Behind the toe, the surface sits between today's and the 1844 surface in
  proportion to the retreat still to come; near the toe it's capped by a perfectly
  plastic snout (thickness sqrt(2 tau x / rho g) at distance x behind the toe), so
  the ice thins to nothing at the front. The other glaciers in view get the same
  fractional change in thickness, inside their present outlines.

Future (today to 2100), every glacier:
  Each year's volume comes from OGGM's projection for that glacier (the median of the
  climate models in the chosen scenario, relative to 2020), applied to today's ice
  volume. The change is spread over the glacier with Huss et al. (2010)'s pattern for
  large glaciers, which puts most of the thinning at the toe. Ice is gone where the
  surface reaches the bed.

    from surfaces import Melt
    m = Melt()            # reads melt/data
    s, ice = m.state(1950.5)
"""
from __future__ import annotations

import json

import numpy as np
import rasterio

import config as C


def _read(name):
    with rasterio.open(C.DATA / name) as s:
        a = s.read(1)
        nd = s.nodata
    return a if nd is None else np.where(a == nd, np.nan, a)


class Melt:
    def __init__(self, scenario: str = C.SCENARIO):
        self.info = json.loads((C.DATA / "place.json").read_text())
        self.dem = _read("dem.tif").astype(np.float64)
        self.bed = _read("bed.tif").astype(np.float64)
        self.lab = _read("glaciers.tif").astype(np.int32)
        self.ice_now = _read("ice_now.tif").astype(bool)
        self.s1844 = _read("s1844.tif").astype(np.float64)
        self.along = _read("along.tif").astype(np.float64)
        self.toe_s = float(self.info["toe_s"])
        self.ath = int(self.info["athabasca_index"])
        self.ground = np.where(self.ice_now, self.bed, self.dem)
        self.h_now = np.where(self.ice_now, self.dem - self.bed, 0.0)
        self.K = np.sqrt(2 * C.TAU / (C.RHO * C.G))
        self.has44 = np.isfinite(self.s1844)
        yrs, frac = zip(*C.RETREAT)
        self._ry, self._rf = np.array(yrs, float), np.array(frac, float)
        # the Athabasca's tongue thickness in 1844 relative to today, for the other glaciers
        ath_now = self.h_now[self.lab == self.ath].sum()
        ath_44 = np.nansum(np.where(self.has44 & (self.lab == self.ath), self.s1844 - self.ground, self.h_now * (self.lab == self.ath)))
        self.past_gain = max(0.0, ath_44 / max(ath_now, 1) - 1)
        self.set_scenario(scenario)

    # ---- future
    def set_scenario(self, scenario):
        self.scenario = scenario
        pr = self.info["projections"][scenario]
        self.p_years = pr["years"]
        gls = self.info["glaciers"]
        ath_id = gls[self.ath - 1]["id"]
        default = pr["glaciers"].get(ath_id, {"median": [1.0] * len(self.p_years)})["median"]
        self.rel = {}
        for i, g in enumerate(gls, start=1):
            r = pr["glaciers"].get(g["id"], {}).get("median", default)
            self.rel[i] = np.array(r, float)
        self._future = {C.TODAY: self.h_now.astype(np.float32)}

    def _rel(self, gi, year):
        y = np.array(self.p_years, float)
        return float(np.interp(year, y, self.rel[gi]) / np.interp(C.TODAY, y, self.rel[gi]))

    def _step(self, h, year):
        """Thickness at the start of `year` from thickness at `year - 1` (Huss delta-h)."""
        h = h.copy()
        a, b, c, g = (C.HUSS[k] for k in ("a", "b", "c", "g"))
        cell = C.RES ** 2
        for gi in np.unique(self.lab[self.lab > 0]):
            if gi == self.ath:
                continue
            m = (self.lab == gi) & (h > 0)
            if not m.any():
                continue
            v0 = (self.h_now * (self.lab == gi)).sum() * cell
            target = v0 * self._rel(gi, year)
            dv = target - h[m].sum() * cell
            if dv >= 0:
                continue
            for _ in range(4):               # cells that run out of ice pass their share on
                m = (self.lab == gi) & (h > 0)
                if not m.any() or dv >= -1:
                    break
                z = self.bed[m] + h[m]
                hr = (z.max() - z) / max(z.max() - z.min(), 1)
                dh = (hr + a) ** g + b * (hr + a) + c
                dh = np.clip(dh, 0, None)
                if dh.sum() <= 0:
                    dh = np.ones_like(dh)
                f = dv / (dh.sum() * cell)
                new = h[m] + f * dh
                lost = np.minimum(new, 0).sum() * cell
                h[m] = np.maximum(new, 0)
                dv = lost
        return h.astype(np.float32)

    def _athabasca(self, year: int):
        """The Athabasca's thickness in a future year: its volume and its front both follow
        OGGM (the front retreats by OGGM's change in glacier length), and between the two
        the ice thins in Huss et al. (2010)'s pattern, solved for the right volume."""
        if not hasattr(self, "_D"):
            a, b, c, g = (C.HUSS[k] for k in ("a", "b", "c", "g"))
            m = (self.lab == self.ath) & (self.h_now > 0)
            z = self.dem[m]
            hr = (z.max() - z) / max(z.max() - z.min(), 1)
            D = np.zeros_like(self.h_now)
            D[m] = np.clip((hr + a) ** g + b * (hr + a) + c, 0.02, None)
            self._D, self._am = D, (self.lab == self.ath)
        gi = self.ath
        cell = C.RES ** 2
        v0 = (self.h_now * self._am).sum() * cell
        target = v0 * self._rel(gi, year)
        s_toe = self.toe_s + self._retreat_future(year)
        h0 = self.h_now * self._am
        def capped(st):
            # the plastic snout from the front at st, over the tongue's last 600 m
            c = np.where(self.along > st, self.K * np.sqrt(np.clip(self.along - st, 0, None)), 0.0)
            return np.where(self.has44 & (self.along - st < 600), c, np.inf)

        def vol(a, cap):
            h = np.minimum(np.clip(h0 - a * self._D, 0, None), cap)
            return h, h.sum() * cell
        cap = capped(s_toe)
        h, v = vol(0.0, cap)
        if v <= target:
            # OGGM's retreat, measured along its own longer flowline, runs past this tongue
            # late in the century; where front and volume disagree, keep the volume and pull
            # the front back only as far as that allows
            lo, hi = self.toe_s, s_toe
            for _ in range(30):
                mid = (lo + hi) / 2
                if vol(0.0, capped(mid))[1] > target:
                    lo = mid
                else:
                    hi = mid
            return vol(0.0, capped(lo))[0]
        lo, hi = 0.0, 2000.0
        for _ in range(40):
            mid = (lo + hi) / 2
            h, v = vol(mid, cap)
            if v > target:
                lo = mid
            else:
                hi = mid
        return vol((lo + hi) / 2, cap)[0]

    def _retreat_future(self, year):
        gid = self.info["glaciers"][self.ath - 1]["id"]
        g = self.info["projections"][self.scenario]["glaciers"].get(gid, {})
        L = g.get("length_m")
        if not L:
            return 0.0
        y = np.array(self.p_years, float)
        return max(0.0, float(np.interp(C.TODAY, y, L) - np.interp(year, y, L)))

    def _future_h(self, year: int):
        if year not in self._future:
            y0 = max(k for k in self._future if k < year)
            h = self._future[y0]
            for y in range(y0 + 1, year + 1):
                h = self._step(h, y)
                if y % 5 == 0 or y == year:
                    self._future[y] = h
        h = self._future[year]
        return np.where(self.lab == self.ath, self._athabasca(year), h).astype(np.float32)

    # ---- past
    def retreat(self, year):
        return float(np.interp(year, self._ry, self._rf)) * self.toe_s

    def _past(self, year):
        s_toe = self.retreat(year)
        w = 1 - s_toe / self.toe_s
        s = np.where(self.has44, self.ground + (1 - w) * (np.where(self.ice_now, self.dem, self.ground) - self.ground)
                     + w * (self.s1844 - self.ground), np.nan)
        # plastic snout behind the toe
        cap = self.ground + self.K * np.sqrt(np.clip(self.along - s_toe, 0, None))
        s = np.where(self.has44, np.minimum(s, cap), np.nan)
        s = np.where(self.has44 & (self.lab == self.ath) & self.ice_now, np.fmax(s, self.dem), s)
        ice = self.has44 & (s > self.ground + 1)
        h = np.where(ice, s - self.ground, 0.0)
        # the other glaciers thicken in step, inside today's outlines
        other = (self.lab > 0) & (self.lab != self.ath) & self.ice_now
        h = np.where(other & ~ice, self.h_now * (1 + self.past_gain * w * 0.6), h)
        # the Athabasca above the reconstructed tongue keeps today's thickness
        rest = (self.lab == self.ath) & self.ice_now & ~self.has44
        h = np.where(rest, self.h_now, h)
        return h

    # ---- any year
    def thickness(self, year: float) -> np.ndarray:
        if year <= C.TODAY:
            return self._past(year)
        y0 = int(np.floor(year))
        h0 = self._future_h(max(y0, C.TODAY))
        if year == y0:
            return h0
        h1 = self._future_h(y0 + 1)
        f = year - y0
        return h0 * (1 - f) + h1 * f

    def state(self, year: float):
        h = self.thickness(year)
        ice = h > 1.0
        s = np.where(ice, self.ground + h, self.dem)
        # ground that today's ice still covers but has melted out: the bed
        s = np.where(self.ice_now & ~ice, self.bed, s)
        return s.astype(np.float32), ice

    def volume(self, year, mask=None) -> float:
        h = self.thickness(year)
        if mask is not None:
            h = h * mask
        return float(h.sum() * C.RES ** 2)
