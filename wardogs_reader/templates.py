"""Fixed HUD labels found by template matching (about 1 ms each).

Templates are grayscale crops taken from 1280x720 cockpit-view frames. Each label
is searched inside its home region only, at a few scales: the heli HUD shrinks to
about 75% in third-person camera, and HUD-scale settings move it a little too.
"""
import os

import cv2

from .regions import REGIONS

HERE = os.path.join(os.path.dirname(__file__), "templates")

# label -> region it lives in
HOME = {
    "collective_lift": "hints",
    "deploy_flares": "hints",
    "change_seat": "hints",
    "spd_label": "gauges",
    "alt_label": "gauges",
    "view_damage_log": "damagelog",
    "give_up": "bottom",
    "call_for_help": "bottom",
    "toggle_legend": "bottom",
    "inventory": "top",
    "equipment_vendor": "top",
    "vendor": "top",
}

SCALES = (1.0, 0.9, 1.1)
# the heli HUD shrinks to ~75% in third-person camera
SMALL = ("collective_lift", "deploy_flares", "change_seat", "spd_label", "alt_label")
SCALES_SMALL = SCALES + (0.8, 0.75, 0.7)


class Templates:
    def __init__(self):
        self.t = {}
        for name in HOME:
            scales = SCALES_SMALL if name in SMALL else SCALES
            # name.png plus optional look variants name_2.png, ... (e.g. third-person gauges)
            for v, suffix in enumerate(["", "_2", "_3"]):
                p = os.path.join(HERE, name + suffix + ".png")
                if os.path.exists(p):
                    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                    self.t.setdefault(name, []).extend(
                        (s, v, cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s != 1 else img)
                        for s in scales)
        self.last = {}                       # name -> best score of the latest check (diagnostics)

    def locate(self, gray, name):
        """Best match of the label inside its home region: (score 0..1, x, y, scale, variant) in frame pixels."""
        if name not in self.t:
            return 0.0, 0, 0, 1.0, 0
        h, w = gray.shape
        x0, y0, x1, y1 = REGIONS[HOME[name]]
        ox, oy = int(x0 * w), int(y0 * h)
        roi = gray[oy:int(y1 * h), ox:int(x1 * w)]
        best = (0.0, 0, 0, 1.0, 0)
        for s, v, tpl in self.t[name]:
            if tpl.shape[0] > roi.shape[0] or tpl.shape[1] > roi.shape[1]:
                continue
            r = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(r)
            if mx > best[0]:
                best = (float(mx), ox + loc[0], oy + loc[1], s, v)
        self.last[name] = round(best[0], 2)
        return best

    def score(self, gray, name):
        return self.locate(gray, name)[0]

    def seen(self, gray, *names, thresh=0.72):
        return any(self.score(gray, n) >= thresh for n in names)
