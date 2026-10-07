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
    "vspd_label": "gauges",           # ground vehicles: SPD box sits a little lower, no ALT box
    "alt_label": "gauges",
    "hammer": "weapon",
    "view_damage_log": "damagelog",
    "give_up": "bottom",
    "call_for_help": "bottom",
    "toggle_legend": "bottom",
    "inventory": "top",
    "equipment_vendor": "top",
    "vendor": "top",
    "rebuy": "rebuy",
    "medkit": "medkit",
}

# The whole game UI changes size between benged's setups (his Twitch stream draws it at ~75%
# of his older Kick VODs), so every label is tried at every size, the last good size first.
SCALES = (1.0, 0.75, 0.9, 0.8, 1.1, 0.7)
# "Stretched res": a 16:10 game (e.g. 1440x900) stretched to a 16:9 screen draws the HUD ~11% wider;
# 4:3 stretched (e.g. 1440x1080) ~33% wider. The reader learns which one his picture uses.
STRETCH = (1.0, 1.111, 1.333)


class Templates:
    def __init__(self):
        self.t = {}
        for name in HOME:
            scales = SCALES
            # name.png plus optional look variants name_2.png, ... (e.g. third-person gauges)
            for v, suffix in enumerate(["", "_2", "_3"]):
                p = os.path.join(HERE, name + suffix + ".png")
                if os.path.exists(p):
                    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                    self.t.setdefault(name, []).extend(
                        (s, v, xs, cv2.resize(img, None, fx=s * xs, fy=s, interpolation=cv2.INTER_AREA)
                         if (s, xs) != (1.0, 1.0) else img)
                        for xs in STRETCH for s in scales)
        self.last = {}                       # name -> best score of the latest check (diagnostics)
        self.pref = {}                       # name -> index of the size/variant that matched last
        self.stretch = None                  # learned sideways stretch of his picture (None = try all)
        self.votes = {}
        self.calls = 0
        # speed: once a label has matched well, only its matching size/look is tried on most calls; the
        # full search (every size x stretch x look, ~10x the work) runs on every 8th call (other setups,
        # settings changes)
        self.known = {}                      # name -> candidate indices that matched >= 0.8
        self.ncalls = {}
        self.profile = "home"
        self._seed()

    def use_profile(self, profile):
        """Which PC he's playing on. 'home' = his own 1440x900 fullscreen (start with the label cuts taken
        from it); anything else (his buddy's 1920x1080 windowed, ...) starts from scratch and learns the
        right label sizes within a few seconds (full search every 4th call until each label is found)."""
        self.profile = profile
        self.pref, self.votes, self.stretch = {}, {}, None
        self.known = {}
        if profile == "home":
            self._seed()

    def _seed(self):
        """Start with the actual-size, unstretched candidates (the _2/_3 looks are cut from benged's own
        1440x900 HUD, so on his PC that's what matches); the full search still runs every 8th call."""
        for name, cands in self.t.items():
            self.known[name] = [i for i, (s, v, xs, _) in enumerate(cands) if s == 1.0 and xs == 1.0][-3:]

    def locate(self, gray, name, stop=None):
        """Best match of the label inside its home region:
        (score 0..1, x, y, scale, variant, horizontal scale) in frame pixels."""
        if name not in self.t:
            return 0.0, 0, 0, 1.0, 0, 1.0
        self.calls += 1
        if self.calls % 6000 == 0:           # every few minutes look at every stretch again (settings change)
            self.stretch, self.votes = None, {}
            self.use_profile(self.profile)
        h, w = gray.shape
        x0, y0, x1, y1 = REGIONS[HOME[name]]
        ox, oy = int(x0 * w), int(y0 * h)
        roi = gray[oy:int(y1 * h), ox:int(x1 * w)]
        best, best_i = (0.0, 0, 0, 1.0, 0, 1.0), None
        cands = self.t[name]
        known = self.known.get(name, [])
        n = self.ncalls[name] = self.ncalls.get(name, 0) + 1
        order = list(known)
        if (not known and n % 4 == 0) or n % 8 == 0:
            first = self.pref.get(name, 0)
            order += [j for j in [first] + list(range(len(cands))) if j not in order]
            order = list(dict.fromkeys(order))
        for i in order:
            s, v, xs, tpl = cands[i]
            if self.stretch is not None and xs != self.stretch:
                continue
            if tpl.shape[0] > roi.shape[0] or tpl.shape[1] > roi.shape[1]:
                continue
            r = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(r)
            if mx > best[0]:
                best, best_i = (float(mx), ox + loc[0], oy + loc[1], s, v, s * xs), i
            if stop is not None and mx >= stop:
                break
        if best_i is not None and best[0] >= 0.7:
            self.pref[name] = best_i
        if best_i is not None and best[0] >= 0.8 and best_i not in known:
            self.known[name] = ([best_i] + known)[:3]
        if best_i is not None and best[0] >= 0.85 and self.stretch is None:
            xs = cands[best_i][2]
            self.votes[xs] = self.votes.get(xs, 0) + 1
            if self.votes[xs] >= 3:          # three confident matches: that's his picture
                self.stretch = xs
        self.last[name] = round(best[0], 2)
        return best

    def locate_near(self, gray, name, x, y, s, v, xs, pad=16):
        """Match one label at a known size/look/stretch inside a small window around (x, y)."""
        for cs, cv, cxs, tpl in self.t.get(name, []):
            if (cs, cv, cxs) == (s, v, xs):
                th, tw = tpl.shape
                x0, y0 = max(0, int(x) - pad), max(0, int(y) - pad)
                roi = gray[y0:int(y) + th + pad, x0:int(x) + tw + pad]
                if roi.shape[0] < th or roi.shape[1] < tw:
                    break
                _, mx, _, loc = cv2.minMaxLoc(cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED))
                return float(mx), x0 + loc[0], y0 + loc[1], s, v, s * xs
        return 0.0, 0, 0, s, v, s * xs

    def score(self, gray, name):
        return self.locate(gray, name)[0]

    def seen(self, gray, *names, thresh=0.72):
        return any(self.locate(gray, n, stop=thresh)[0] >= thresh for n in names)
