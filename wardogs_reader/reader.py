"""Turns WARDOGS screenshots into overlay events.

Feed it BGR frames (any 16:9 size, resized to 1280x720) with a timestamp in seconds:

    r = Reader(names=["Benged"])
    for ev in r.process(frame, t): ...

Events are plain dicts with a "type":
  state     {pilot, vehicle, scoped, hurt, screen}   screen = none|down|map|vendor|menu|loading
  kill      {dist, victim, via}                      via = popup | feed | popup+feed
  selfkill  {pilot}                                  his name on both sides (heli crash / self-nade)
  down      {killer}                                 knocked down (VIEW DAMAGE LOG / GIVE UP)
  up        {}                                       back up without respawning (revived)
  died      {}                                       down ended on the map / loading screen
  xp        {kind, amount, money, text}
  cash      {match, wallet}
  pilot     {spd, agl, hdg}
  killed_by {killer, dist}                           his death line in the kill feed
  line      {region, text}                           unrecognised feed text (to discover new events)

Fixed labels (pilot controls, damage log, vendor tabs ...) are template-matched,
which costs ~1 ms. OCR only runs on text that changes, and each feed line is
read once (see CachedLines).
"""
import re
import difflib
from collections import Counter

import cv2
import numpy as np

from .regions import crop
from .ocr import read_line, read_lines, norm, stroke_mask, CachedLines
from .templates import Templates

IGNORE = ("ROTARYVEHICLE", "OCCUPIEDSEATS", "LAKOTA", "UNLOCKED")
XP_RULES = [
    ("SUPPLIESDELIVERED", "supplies"),
    ("SUPPLIESDEPOSITED", "supplies"),
    ("CONTROLZONEENTERED", "zone_enter"),
    ("CONTROLZONEPRESENCE", "zone"),
    ("VEHICLEREFUELLING", "refuel"),
    ("VEHICLEREPAIRING", "repair"),
    ("WAITEDTOBEREVIVED", "revived_me"),
    ("BUILDINGCOMPLETE", "build_complete"),
    ("BUILDABLEDESTROYED", "buildable_destroyed"),
    ("BUILDING", "build"),
    ("REVIVEDTEAMMATE", "revive"),
    ("REVIVEDPLAYER", "revive_assist"),
    ("PLAYERASSIST", "assist"),
    ("TARGETSPOTTED", "spot"),
    ("TIP", "tip"),
    ("LEVEL", "level"),
]
MONEY_RE = re.compile(r"(-)?\s*[S$]\s?(\d{1,3}(?:[.,]\d{3})+|\d{1,7})")
DIST_RE = re.compile(r"\[?\s*(\d{1,3})\s*[mM]\s*\]?")
XP_RE = re.compile(r"(\d{1,4})\s*XP", re.I)


def _digits(s):
    """OCR letter/digit swaps inside numbers: I/l->1, O->0, B->8, S->5."""
    return s.translate(str.maketrans({"I": "1", "l": "1", "O": "0", "o": "0", "B": "8", "S": "5"}))


# a number as OCR tends to read it: digits or look-alikes, with ".S00"-style thousands groups
_NUM = r"[\dIlOoB](?:[\dIlOoB]|[.,][\dIlOoBS][\dIlOoB]{2})*"
_MONEY_FIX = re.compile(r"(?<![A-Za-z])S(" + _NUM + r")(?![A-Za-z])")
_XP_FIX = re.compile(r"(?<![A-Za-z])([\dSOoIlB]{1,4})\s*XP", re.I)


def _fix_money_text(raw):
    """'+SI60' -> '+$160', 'S2.S00' -> '$2.500', '-S7.800 S1.354.694' -> '-$7.800 $1.354.694'."""
    return _MONEY_FIX.sub(lambda m: "$" + _digits(m.group(1)), raw)


def _fix_xp_text(raw):
    """'SOOXP' -> '500XP', '18OXP' -> '180XP'."""
    return _XP_FIX.sub(lambda m: _digits(m.group(1)) + "XP", raw)


def classify(k):
    """Map a normalised feed line to an event kind, tolerating a few misread letters."""
    for p, v in XP_RULES:
        if p in k:
            return v
    for p, v in XP_RULES:
        L = len(p)
        if L >= 8:
            for i in range(0, max(1, len(k) - L + 1)):
                if difflib.SequenceMatcher(None, k[i:i + L], p).ratio() >= 0.78:
                    return v
    return None


def _money(sign, num):
    v = int(re.sub(r"[.,]", "", num))
    return -v if sign else v


class LineTracker:
    """Emits each newly-appeared feed line once.

    A line counts once it is read in two consecutive passes; we emit when the
    count of a normalised key on screen goes up. Keys not seen for `hold`
    seconds are forgotten, so a repeat of the same event later fires again.
    """

    def __init__(self, hold=6.0):
        self.prev = Counter()
        self.confirmed = Counter()
        self.last_seen = {}
        self.hold = hold

    def _canon(self, k):
        """Fold OCR variants of the same line onto one key."""
        if k in self.last_seen:
            return k
        for old in self.last_seen:
            if abs(len(old) - len(k)) <= 3 and difflib.SequenceMatcher(None, old, k).ratio() >= 0.8:
                return old
        return k

    def update(self, keys, t):
        cur = Counter(self._canon(k) for k in keys)
        stable = cur & self.prev
        new = []
        for k, n in stable.items():
            if t - self.last_seen.get(k, -999) > self.hold:
                self.confirmed[k] = 0
            if n > self.confirmed[k]:
                new += [k] * (n - self.confirmed[k])
                self.confirmed[k] = n
            self.last_seen[k] = t
        self.prev = cur
        return new


class Reader:
    def __init__(self, names=("Benged",), mask=()):
        self.names = [norm(n) for n in names if n]
        self.mask = mask                      # [(x0,y0,x1,y1) fractions] blanked before reading (VOD tests: webcam)
        self.tpl = Templates()
        self.t = 0.0
        self.next = {}
        self.xp_lines = CachedLines(max_w=440)
        self.kf_lines = CachedLines(max_w=470, max_new=3)
        self.xp = LineTracker()
        self.kf = LineTracker(hold=7.0)
        self.state = dict(pilot=False, vehicle=False, scoped=False, hurt=False, screen="none")
        self._pilot_last = self._vehicle_last = -999
        self._cash = self._cash_cand = None
        self._cash_miss = 0
        self._popup_last = -999
        self._down_since = None
        self._screen_votes = []
        self._pending_kill = None
        self._hurt_frames = 0
        self._gauges = None                   # (SPD label, ALT label) matches while the heli gauges are on screen
        self._spd = self._agl = None
        self._jump = {}

    def _selfkill(self, ev, **kw):
        if self.t - getattr(self, "_selfkill_last", -999) > 20:
            self._selfkill_last = self.t
            ev.append(dict(type="selfkill", **kw))

    def _due(self, key, every):
        if self.t >= self.next.get(key, -1):
            self.next[key] = self.t + every
            return True
        return False

    def _is_me(self, n):
        for name in self.names:
            if not name:
                continue
            if name in n:
                return True
            L = len(name)
            for i in range(0, max(1, len(n) - L + 1)):
                if difflib.SequenceMatcher(None, n[i:i + L], name).ratio() >= 0.8:
                    return True
        return False

    # ------------------------------------------------------------------ main
    def process(self, frame, t):
        self.t = t
        ev = []
        if frame.shape[1] != 1280:
            frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
        if self.mask:
            frame = frame.copy()
            for x0, y0, x1, y1 in self.mask:
                frame[int(y0 * 720):int(y1 * 720), int(x0 * 1280):int(x1 * 1280)] = 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_state = dict(self.state)

        if self._due("screen", 0.5):
            self._read_screens(frame, gray, ev)
        in_game = self.state["screen"] == "none"

        if self._due("hints", 0.5):
            # pilot seat: the pilot-only control hints, or the SPD + ALT gauges
            # (either is enough; the hints can be toggled off)
            self._gauges = self._find_gauges(gray)
            if self._gauges or self.tpl.seen(gray, "collective_lift", "deploy_flares"):
                self._pilot_last = t
            elif self.tpl.seen(gray, "change_seat"):
                self._vehicle_last = t
        if in_game:
            if self._due("cash", 1.0):
                self._read_cash(frame, ev)
            if self._due("xp", 0.5):
                self._read_xp(frame, ev)
            if self._due("popup", 0.25):
                self._read_popup(frame, ev)
            if self._gauges and self._due("gauges", 1.0):
                self._read_gauges(frame, ev)
            # damage vignette must hold for 2 frames
            self._hurt_frames = self._hurt_frames + 1 if self._vignette(frame) > 34 else 0
            self.state["hurt"] = self._hurt_frames >= 2
            self.state["scoped"] = self._scope(gray)
        if self._due("kf", 0.5):
            self._read_killfeed(frame, ev)

        self.state["pilot"] = t - self._pilot_last < 4.0
        self.state["vehicle"] = (not self.state["pilot"]) and t - self._vehicle_last < 4.0
        self._flush_kill(ev)
        if self.state != prev_state:
            ev.append(dict(type="state", **self.state))
        for e in ev:
            e.setdefault("t", round(t, 2))
        return ev

    # ---------------------------------------------------------------- screens
    def _read_screens(self, frame, gray, ev):
        screen = "none"
        if self.tpl.seen(gray, "vendor", "equipment_vendor", thresh=0.82):
            screen = "vendor"                      # EQUIPMENT / VEHICLE / GARAGE VENDOR
        elif self.tpl.seen(gray, "inventory", thresh=0.82):
            screen = "menu"                        # INVENTORY / SCOREBOARD / PROGRESSION tabs
        elif self.tpl.seen(gray, "view_damage_log", "give_up", "call_for_help"):
            screen = "down"
        elif self.tpl.seen(gray, "toggle_legend"):
            screen = "map"
        elif self._cash_miss >= 2 and self._due("loading", 2.0):
            ld = " ".join(norm(x[0]) for x in read_lines(crop(frame, "loading")))
            if "WARDOGS" in ld:
                screen = "loading"
        self._screen_votes = (self._screen_votes + [screen])[-2:]
        if len(self._screen_votes) == 2 and self._screen_votes[0] == self._screen_votes[1]:
            self._set_screen(screen, frame, ev)

    def _set_screen(self, screen, frame, ev):
        old = self.state["screen"]
        if screen == old:
            return
        if screen == "down" and self._down_since is None:
            self._down_since = self.t
            killer = self._killer_from_log(frame)
            self_inflicted = bool(killer) and self._is_me(killer)
            ev.append(dict(type="down", killer="" if self_inflicted else killer, self_inflicted=self_inflicted))
            if self_inflicted:
                self._selfkill(ev, pilot=self.t - self._pilot_last < 12, via="damagelog")
        elif self._down_since is not None and screen == "none" and old in ("down", "map"):
            ev.append(dict(type="up", after=round(self.t - self._down_since, 1)))
            self._down_since = None
        elif self._down_since is not None and screen in ("loading", "map"):
            ev.append(dict(type="died"))
            self._down_since = None
        self.state["screen"] = screen

    def _killer_from_log(self, frame):
        names = []
        for txt, conf, _ in read_lines(crop(frame, "damagelog")):
            n = norm(txt)
            if "VIEWDAMAGELOG" in n or "NEARBY" in n:
                continue
            m = re.match(r"^\d{2,3}(.+)$", n)       # "025QHEYBHIP": damage, then name
            if m and len(m.group(1)) >= 3:
                names.append(m.group(1))
        return Counter(names).most_common(1)[0][0] if names else ""

    # ------------------------------------------------------------- HUD text
    def _read_cash(self, frame, ev):
        text, _ = read_line(crop(frame, "cash"))
        text = _fix_money_text(text)
        vals = [_money(s, n) for s, n in MONEY_RE.findall(text)]
        if len(vals) < 2:
            self._cash_miss += 1
            return
        self._cash_miss = 0
        cand = (vals[-2], vals[-1])
        self._cash_hist = (getattr(self, "_cash_hist", []) + [cand])[-5:]
        best, n = Counter(self._cash_hist).most_common(1)[0]
        if n >= 3 and best != self._cash:
            if self._cash is None or abs(best[1] - self._cash[1]) < 60000:
                self._cash = best
                ev.append(dict(type="cash", match=best[0], wallet=best[1]))

    def _read_xp(self, frame, ev):
        lines = self.xp_lines.read(crop(frame, "xpfeed"))
        keys = [norm(x[0]) for x in lines if x[1] > 0.6 and len(norm(x[0])) >= 4]
        for k in self.xp.update(keys, self.t):
            if any(i in k for i in IGNORE):
                continue
            raw = next((x[0] for x in lines if norm(x[0]) == k), k)
            raw = _fix_xp_text(_fix_money_text(raw))
            kind = classify(k)
            if kind is None:
                ev.append(dict(type="line", region="xpfeed", text=raw))
                continue
            xp = XP_RE.search(raw)
            money = MONEY_RE.search(raw) if not xp else None
            ev.append(dict(type="xp", kind=kind, amount=int(xp.group(1)) if xp else 0,
                           money=_money(*money.groups()) if money else 0, text=raw))

    def _read_popup(self, frame, ev):
        img = crop(frame, "popup")
        # cheap gate: the popup is a short block of bright strokes; skip OCR otherwise
        if int(stroke_mask(img).sum()) < 120 or self.t - self._popup_last < 2.0:
            return
        txt = " ".join(norm(x[0]) for x in read_lines(img))
        if "CONFIRM" in txt or "2000" in txt:
            self._popup_last = self.t
            self._kill_signal("popup", None, "")

    def _read_killfeed(self, frame, ev):
        lines = self.kf_lines.read(crop(frame, "killfeed"))
        keys = [norm(x[0]) for x in lines if x[1] > 0.5]
        for k in self.kf.update(keys, self.t):
            if not self._is_me(k):
                continue
            raw = next((x[0] for x in lines if norm(x[0]) == k), k)
            d = DIST_RE.search(raw)
            dist = int(d.group(1)) if d else None
            left, right = (raw[:d.start()], raw[d.end():]) if d else (raw[:len(raw) // 2], raw[len(raw) // 2:])
            me_l, me_r = self._is_me(norm(left)), self._is_me(norm(right))
            if me_l and me_r:
                self._selfkill(ev, pilot=self.t - self._pilot_last < 12, via="killfeed", text=raw)
            elif me_l:
                self._kill_signal("feed", dist, right.strip(" []|"))
            else:
                # his own death line: "[2EZ]brave88 [11m] [KA]Benged"
                ev.append(dict(type="killed_by", killer=left.strip(" []|"), dist=dist, text=raw))

    def _find_gauges(self, gray, thresh=0.7):
        sp = self.tpl.locate(gray, "spd_label")
        if sp[0] < thresh:
            return None
        al = self.tpl.locate(gray, "alt_label")
        if al[0] < thresh:
            return None
        if not (0 < al[1] - sp[1] < 330 and abs(al[2] - sp[2]) < 8):    # ALT sits right of SPD, same row
            return None
        return sp, al

    @staticmethod
    def _gauge_number(text, after):
        """'KM/H 123' -> 123, 'AGL1S' -> 15. Takes what follows the unit; fixes look-alike letters."""
        u = text.upper()
        i = max(u.rfind(c) for c in after)
        tail = u[i + 1:] if i >= 0 else u
        tail = tail.translate(str.maketrans("OQDIL|SBGZ", "0001115862"))
        d = re.findall(r"\d+", tail)
        return int(d[-1]) if d else None

    def _steady(self, key, v, max_step):
        """Drop one-off OCR jumps (a lost digit: 242 -> 24); accept a jump the next reading agrees with."""
        last = getattr(self, "_" + key)
        if v is None:
            return last
        pend = self._jump.get(key)
        if last is None or abs(v - last) <= max_step or (pend is not None and abs(v - pend) <= max_step):
            self._jump.pop(key, None)
            setattr(self, "_" + key, v)
            return v
        self._jump[key] = v
        return last

    # number line under each label, relative to the label match: (top, bottom, right) in template pixels
    # per template variant (0 = cockpit view, 1 = third-person view: narrower boxes, smaller text)
    GAUGE_LINE = {("spd", 0): (14, 29, 72), ("spd", 1): (14, 29, 55),
                  ("alt", 0): (14, 29, 66), ("alt", 1): (14, 29, 54)}

    def _gauge_text(self, frame, key, match):
        _, x, y, s, v = match
        top, bot, right = self.GAUGE_LINE[(key, v)]
        return read_line(frame[int(y + top * s):int(y + bot * s), max(0, x - 2):int(x + right * s)])[0]

    def _read_gauges(self, frame, ev):
        s = self._gauge_text(frame, "spd", self._gauges[0])
        a = self._gauge_text(frame, "alt", self._gauges[1])
        c, _ = read_line(crop(frame, "compass"))
        m = re.search(r"(\d{3})\s*(NE|NW|SE|SW|N|E|S|W)\b", c)
        spd = self._gauge_number(s, "H/")
        agl = self._gauge_number(a, "L")
        spd = self._steady("spd", spd if spd is not None and spd <= 450 else None, 60)
        agl = self._steady("agl", agl if agl is not None and agl <= 3000 else None, 50)
        ev.append(dict(type="pilot", spd=spd, agl=agl, hdg=int(m.group(1)) % 360 if m else None, raw=[s, a]))

    # ------------------------------------------- kill fusion (popup + feed line)
    def _kill_signal(self, via, dist, victim):
        p = self._pending_kill
        if p and self.t - p["t0"] < 3.0 and via not in p["via"]:
            p["via"].append(via)
            p["dist"] = p["dist"] or dist
            p["victim"] = p["victim"] or victim
        else:
            self._pending_kill = dict(t0=self.t, via=[via], dist=dist, victim=victim)

    def _flush_kill(self, ev):
        p = self._pending_kill
        if p and (len(p["via"]) == 2 or self.t - p["t0"] >= 1.25):
            ev.append(dict(type="kill", dist=p["dist"], victim=p["victim"], via="+".join(p["via"])))
            self._pending_kill = None

    # ------------------------------------------------------- pixel metrics
    @staticmethod
    def _vignette(frame):
        b = 50
        edge = np.concatenate([frame[:b:2, ::4].reshape(-1, 3), frame[-b::2, ::4].reshape(-1, 3),
                               frame[::4, :b:2].reshape(-1, 3), frame[::4, -b::2].reshape(-1, 3)]).astype(np.int16)
        return float(np.mean(edge[:, 2] - np.maximum(edge[:, 0], edge[:, 1])))

    @staticmethod
    def _scope(g):
        h, w = g.shape
        cs = [g[int(h*.30):int(h*.40), int(w*.22):int(w*.27)], g[int(h*.30):int(h*.40), int(w*.73):int(w*.78)],
              g[int(h*.62):int(h*.72), int(w*.22):int(w*.27)], g[int(h*.62):int(h*.72), int(w*.73):int(w*.78)]]
        centre = g[int(h*.45):int(h*.55), int(w*.45):int(w*.55)]
        return max(float(c.mean()) for c in cs) < 14 and float(centre.mean()) > 40
