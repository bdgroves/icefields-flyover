"""Settings for the Athabasca Glacier melt film: one fixed view from the Icefield Centre
lookout on the Icefields Parkway, 1844 to 2100."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "out"

CRS = "EPSG:32611"          # UTM 11N
RES = 10.0                  # metres; the terrain is resampled from Canada's 30 m MRDEM
# The glacier, its forefield, the lookout and the peaks around it (west, south, east, north)
BBOX = (-117.37, 52.125, -117.15, 52.255)

ATHABASCA = "RGI60-02.12441"
ICEFIELD_CENTRE = (52.2197, -117.2244)     # GeoVistas (Gadd 2011): N52 13.179, W117 13.465, 1,975 m
LIA_YEAR = 1844                             # Little Ice Age maximum (Gadd 2011)
EYE_BACK_M = 300                            # camera this far down-valley of the 1844 front...

EYE_UP_M = 150                              # ...and this far above it: a drone over the lookout
# 1844 thickening above the moraine data: fades from its value at the toe to this many
# metres, over this distance up-glacier (calibrated against the 1919-79 map loss)
# At 30 m the terrain can't reliably separate the moraine crests from the valley walls, so the
# 1844 surface uses the published height instead: the east lateral moraine stands about 124 m
# above the valley floor near the toe (Bath Geological Society, 1985), and today's toe is
# roughly 20 m above the floor, so about 104 m more ice at today's toe in 1844.
DETECT_MORAINES = False
DZ_TOE_M = 104.0
UP_MIN_M = 25.0
UP_FADE_M = 5000.0

# How far the toe had retreated from its 1844 position (metres), from:
#   Gadd (2011): the 1844 front reached where Icefield Centre is now, and the glacier
#     is "nearly 2 km shorter" today; retreat began in the mid-19th century.
#   Bath Geological Society (1985): about 1.5 km from 1870 to 1981.
#   Reynolds & Young (1997): about 1 km from 1919 to 1979.
#   Recent retreat about 5 m a year (Mountain Legacy Project, 2020); 10-25 m a summer (Gadd).
# The total is scaled to the measured distance from the 1844 position to today's toe.
RETREAT = [(1844, 0.0), (1870, 0.0), (1919, 0.31), (1979, 0.83), (1981, 0.85), (2011, 0.96), (2025, 1.0)]

TODAY = 2025
SCENARIO = "ssp245"                     # the film's future; the end card shows all three
SCENARIOS = ["ssp126", "ssp245", "ssp585"]
SCENARIO_NAMES = {"ssp126": "SSP1-2.6 (low emissions)", "ssp245": "SSP2-4.5 (middle)",
                  "ssp585": "SSP5-8.5 (high emissions)"}

# Huss et al. (2010) elevation-change pattern for large glaciers: most thinning at the toe.
HUSS = dict(a=-0.02, b=0.12, c=0.0, g=6.0)
# Perfectly plastic ice (Nye): a snout's thickness grows as sqrt(2 tau x / (rho g)).
TAU, RHO, G = 100e3, 900.0, 9.81
SNOUT_R_M = 450.0           # how rounded the snout is, seen from above

# Morning light from the east-south-east on 20 August, 10:30 a.m. MDT, so the glacier,
# which faces north-east toward the lookout, is lit from the front.
SUN_UTC = (2026, 8, 20, 16, 30, 0)
SUN_LATLON = (52.2, -117.25)

LABELS = {   # name: (lat, lon, label)  (Wikipedia infoboxes)
    "athabasca": (52.18056, -117.20167, "Mount Athabasca"),
    "andromeda": (52.17389, -117.23611, "Mount Andromeda"),
    "snowdome": (52.18694, -117.31694, "Snow Dome"),
    "kitchener": (52.21611, -117.32083, "Mount Kitchener"),
}


def place() -> dict:
    p = DATA / "place.json"
    return json.loads(p.read_text()) if p.exists() else {}
