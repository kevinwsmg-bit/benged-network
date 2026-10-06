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
from .ocr import read_line, read_lines, read_boxes, norm, stroke_mask, CachedLines
from .templates import Templates
from .feed import Feed

IGNORE = ("ROTARYVEHICLE", "OCCUPIEDSEATS", "LAKOTA", "UNLOCKED", "STORAGE", "BACKPACK", "SQUAD", "VEHICLEINVENTORY")
XP_RULES = [
    ("SUPPLIESDELIVERED", "supplies"),
    ("SUPPLIESDEPOSITED", "supplies"),
    ("CONTROLZONEENTERED", "zone_enter"),
    ("CONTROLZONEPRESENCE", "zone"),
    ("HOTZONEPRESENCE", "zone"),
    ("DRILLPRESENCE", "zone"),
    ("PASSENGERSURVIVED", "passenger"),     # pilot: each passenger dropped off alive (+$500)
    ("TACTICALDEPLOYMENT", "deploy"),       # pilot: troops deployed from his heli
    ("HEALEDTEAMMATE", "heal"),
    ("FOBSUPPLIED", "fob_supplied"),
    ("KILLASSIST", "kill_assist"),
    ("WHEELSDESTROYED", "wheels"),
    ("BRIBE", "bribe"),
    ("AIRTIMEBONUS", "airtime"),            # vehicle jump
    ("VEHICLEREFUELLING", "refuel"),
    ("VEHICLEREPAIRING", "repair"),
    ("WAITEDTOBEREVIVED", "revived_me"),
    ("BUILDINGCOMPLETE", "build_complete"),
    ("BUILDABLEDESTROYED", "buildable_destroyed"),
    ("BUILDING", "build"),
    ("REVIVEDTEAMMATE", "revive"),
    ("REVIVEDPLAYER", "revive_assist"),
    ("PLAYERASSIST", "assist"),
    ("HEADSHOTKILL", "headshot"),            # kill bonus lines (the kill itself is the popup)
    ("SHUTDOWNKILL", "shutdown"),           # killed someone on a streak
    ("SPOTTEDTARGETDESTROYED", "spot_destroy"),   # recon: a target he spotted got killed
    ("SPOTTEDVEHICLEDESTROYED", "spot_destroy"),
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
    if "TEAMKILL" in k:                      # "TEAM KILL ASSIST -$909": a penalty, not an assist
        return "teamkill"
    # "KILL +$1,500" -> "KILL 250XP"; OCR adds letters around it ("KILSI", "YAJKILLSI")
    if re.search(r"KIL[A-Z]{0,3}$", k) and len(k) <= 10 and not any(x in k for x in ("ASSIST", "HEADSHOT", "SHUTDOWN")):
        return "kill"
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
    def __init__(self, names=("Benged",), mask=(), feed_sx=None):
        self.names = [norm(n) for n in names if n]
        self.feed_sx = feed_sx                # sideways stretch of the picture (None: learn it)
        self.mask = mask                      # [(x0,y0,x1,y1) fractions] blanked before reading (VOD tests: webcam)
        self.tpl = Templates()
        self.t = 0.0
        self.next = {}
        self.feed = Feed(classify)
        self.feed.sx = feed_sx or 1.0
        self.cf_lines = CachedLines(max_w=330, max_new=5)
        self._cf, self._cf_raw, self._cf_seen = [], None, -999
        self._cf_total, self._cf_tcand, self._cf_kind = 0, None, None
        self.kf_lines = CachedLines(max_w=470, max_new=2)
        self.pop_lines = CachedLines(max_new=3)
        self.seat_lines = CachedLines(max_new=3)
        self._kf_seen = []                    # kill-feed lines with his name now on screen
        self.state = dict(pilot=False, vehicle=False, hammer=False, medkit=False, scoped=False, hurt=False, screen="none")
        self._pilot_last = self._vehicle_last = self._hammer_last = self._medkit_last = -999
        self._cash = self._cash_cand = None
        self._cash_miss = 0
        self._pop, self._pop_seen = None, -999   # the kill popup on screen now
        self._down_since = None
        self._screen_votes = []
        self._kills, self._kills_ready = [], []   # recent kills (to merge popup + feed), events to send
        self._hurt_frames = 0
        self._gauges = None                   # (SPD label, ALT label) matches while the heli gauges are on screen
        self._vgauge = None                   # ground-vehicle SPD box match
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
        self._full = frame                    # full size: the reward feed is read from this
        if frame.shape[0] != 720:
            # the game sizes its HUD by screen height: scaling every picture to 720 tall keeps the HUD
            # the same size whatever the shape (16:9 -> 1280x720, 16:10 1440x900 -> 1152x720)
            frame = cv2.resize(frame, (round(frame.shape[1] * 720 / frame.shape[0]), 720),
                               interpolation=cv2.INTER_AREA)
        if self.mask:
            frame = frame.copy()
            for x0, y0, x1, y1 in self.mask:
                frame[int(y0 * 720):int(y1 * 720), int(x0 * frame.shape[1]):int(x1 * frame.shape[1])] = 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_state = dict(self.state)

        if self._due("screen", 0.5):
            self._read_screens(frame, gray, ev)
        in_game = self.state["screen"] == "none"

        if self._due("hints", 0.5):
            # pilot seat: the pilot-only control hints, or the SPD + ALT gauges
            # (either is enough; the hints can be toggled off)
            self._gauges = self._find_gauges(gray)
            self._vgauge = None if self._gauges else self._find_vehicle_gauge(gray)
            heli_keys = self.tpl.seen(gray, "collective_lift", "deploy_flares", thresh=0.8)
            seat_keys = self.tpl.seen(gray, "change_seat", thresh=0.8)   # every seat, passengers too
            if self._gauges or heli_keys:
                self._pilot_last = t
            elif self._vgauge or seat_keys:          # driver or passenger (benged: passengers count too)
                self._vehicle_last = t
            elif self.tpl.seen(gray, "hammer", thresh=0.75):    # building tool in hand
                self._hammer_last = t
            elif self.tpl.seen(gray, "medkit", thresh=0.78):    # medkit in hand (REVIVE FRIENDLY / HEAL SELF)
                self._medkit_last = t
            # out of the vehicle: no gauges and no control list for 3 checks in a row (1.5 s), in game
            # (the list is on screen in every seat, and never on foot) -> drop pilot/vehicle now
            # which detectors fired (sent when it changes; only for the session log, to check streams later)
            sig = "".join(c for c, on in (("G", self._gauges), ("V", self._vgauge), ("H", heli_keys),
                                          ("S", seat_keys), ("B", t - self._hammer_last < 0.1),
                                          ("R", t - self._medkit_last < 0.1)) if on)
            if sig != getattr(self, "_sig", None):
                self._sig = sig
                ev.append(dict(type="sig", s=sig or "-"))
            if self._gauges or self._vgauge or heli_keys or seat_keys or self.state["screen"] != "none":
                self._out = 0
            else:
                self._out = getattr(self, "_out", 0) + 1
                if self._out >= 3:                   # 1.5 s
                    self._pilot_last = self._vehicle_last = -999
        if self.state["screen"] == "vendor" and self._due("vcash", 0.5):
            self._read_cash(frame, ev, need=2)          # purchases = wallet going down in the store
            self._read_cart(frame, ev)
        if in_game:
            if self._due("cash", 1.0):
                self._read_cash(frame, ev)
            if self._due("xp", 0.5):
                self._read_xp(frame, ev)
            # (the bottom reward list isn't read any more: the top-right feed carries the same rewards)
            if self._due("popup", 0.5):
                self._read_popup(frame, ev)
            if (self.state["vehicle"] or self.state["pilot"]) and self._due("seats", 4.0):
                self._read_seats(frame, ev)
            if self._gauges and self._due("gauges", 1.0):
                self._read_gauges(frame, ev)
            elif self._vgauge and self._due("gauges", 1.0):
                self._read_vehicle_speed(frame, ev)
            # damage vignette must hold for 2 frames
            self._hurt_frames = self._hurt_frames + 1 if self._vignette(frame) > 34 else 0
            self.state["hurt"] = self._hurt_frames >= 2
            self.state["scoped"] = self._scope(gray)
        if self._due("kf", 0.75):
            self._read_killfeed(frame, ev)

        self.state["pilot"] = t - self._pilot_last < 4.0
        self.state["vehicle"] = (not self.state["pilot"]) and t - self._vehicle_last < 4.0
        self.state["hammer"] = not (self.state["pilot"] or self.state["vehicle"]) and t - self._hammer_last < 4.0
        self.state["medkit"] = (not (self.state["pilot"] or self.state["vehicle"] or self.state["hammer"])
                                and t - self._medkit_last < 3.0)
        self._flush_kill(ev)
        if self.state != prev_state:
            ev.append(dict(type="state", **self.state))
        for e in ev:
            e.setdefault("t", round(t, 2))
        return ev

    # ---------------------------------------------------------------- screens
    def _read_screens(self, frame, gray, ev):
        screen = "none"
        if self.tpl.seen(gray, "vendor", "equipment_vendor", "rebuy", thresh=0.82):
            screen = "vendor"                      # EQUIPMENT / VEHICLE / GARAGE VENDOR
        elif self.tpl.seen(gray, "inventory", thresh=0.82):
            screen = "menu"                        # INVENTORY / SCOREBOARD / PROGRESSION tabs
        elif self.tpl.seen(gray, "view_damage_log", "give_up", "call_for_help"):
            screen = "down"
        elif self.tpl.seen(gray, "toggle_legend"):
            screen = "map"
        elif self._cash_miss >= 2 and self._due("loading", 4.0):
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
    _MONEY_BOX = re.compile(r"^(-?)\s*[$S]\s*(\d{1,3}(?:[.,]\d{3})+|\d{1,7})$")

    def _money_boxes(self, frame):
        """(match, wallet) from the money boxes at the top right: [-$9,203 v][$1,842,031] (+ [125] on
        menus/vendors). Each box is found and read on its own, so the two amounts never run together.
        The match box's minus sign sometimes doesn't read; its red marker means negative."""
        h, w = frame.shape[:2]
        roi = frame[0:int(0.055 * h), int(0.72 * w):w]
        # the boxes rarely change: skip the (slow) box finder while the strip looks the same as last time
        small = cv2.resize(roi, (96, 10), interpolation=cv2.INTER_AREA).astype(np.int16)
        last = getattr(self, "_mb_last", None)
        if last is not None and last[0].shape == small.shape and float(np.abs(small - last[0]).mean()) < 2.0:
            return last[1]
        up = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        vals = []
        for x, t, c in read_boxes(up):
            u = t.upper().replace(" ", "").translate(str.maketrans("OQDIL|", "000111"))
            m = self._MONEY_BOX.match(u)
            if m and c > 0.5:
                vals.append((x / 2, m.group(1) == "-", int(re.sub(r"[.,]", "", m.group(2)))))
        if len(vals) < 2:
            self._mb_last = (small, None)
            return None
        (x0, neg, match), (x1, _, wallet) = vals[-2], vals[-1]
        if not neg:
            seg = roi[:, int(x0):int(x1)]
            b, g, r = (seg[:, :, i].astype(int) for i in range(3))
            neg = int(((r > 150) & (g < 90) & (b < 90)).sum()) > 12
        self._mb_last = (small, ((-match if neg else match), wallet))
        return self._mb_last[1]

    def _read_cart(self, frame, ev):
        """Vendor cart total from the green Purchase button ("$350"); two matching reads."""
        amt = 0
        for _, t, c in read_boxes(crop(frame, "cart")):
            u = t.upper().replace(" ", "").translate(str.maketrans("OQDIL|", "000111"))
            m = self._MONEY_BOX.match(u)
            if m and c > 0.6:
                amt = int(re.sub(r"[.,]", "", m.group(2)))
                break
        if amt == getattr(self, "_cart_cand", None) and amt != getattr(self, "_cart", None):
            self._cart = amt
            ev.append(dict(type="cart", amount=amt))
        self._cart_cand = amt

    def _read_cash(self, frame, ev, need=3):
        """Match money + wallet. In game: 3 of the last 5 reads must agree. At a vendor (need=2) it
        reads faster so a purchase registers right away."""
        mw = self._money_boxes(frame)
        if mw is None:
            self._cash_miss += 1
            return
        self._cash_miss = 0
        self._cash_hist = (getattr(self, "_cash_hist", []) + [mw])[-5:]
        best, n = Counter(self._cash_hist).most_common(1)[0]
        if n >= need and best != self._cash:
            if self._cash is None or abs(best[1] - self._cash[1]) < 60000:
                self._cash = best
                ev.append(dict(type="cash", match=best[0], wallet=best[1]))

    _LOOK = str.maketrans("OoQDIl|SZB", "0000111528")
    _ENTRY_XP = re.compile(r"([0-9OoQDIl|SZB]{2,4})\s*X\s*P\W*$")
    _ENTRY_MONEY = re.compile(r"[$S]\s*([0-9OoIlZB][0-9OoIlZB.,]{1,6})\W*$")

    @classmethod
    def _feed_entry(cls, raw):
        """'PASSENGER SURVIVED 100XP' -> (kind, 'xp', 100); '... +$500' -> (kind, '$', 500).
        Reads the amount from the end of the line only (OCR often glues it to the words).
        None for the '+$2,500' running total and other non-entries."""
        n = norm(raw)
        if len(re.sub(r"[^A-Z]", "", n)) < 8:
            return None
        kind = classify(n) or re.sub(r"[^A-Z]", "", n)[:10]
        u = raw.upper()
        m = cls._ENTRY_XP.search(u)
        if m:
            v = m.group(1).translate(cls._LOOK)
            return kind, "xp", int(v) if v.isdigit() else 0
        m = cls._ENTRY_MONEY.search(u)
        if m:
            v = re.sub(r"[.,]", "", m.group(1)).translate(cls._LOOK)
            return kind, "$", int(v) if v.isdigit() else 0
        return kind, "?", 0

    # XP paid per $ for rewards whose money goes into the boxed running total (measured on
    # benged's clips: PASSENGER SURVIVED $500 = 100XP and $1,000 = 200XP; TACTICAL DEPLOYMENT
    # $195 = 50XP and $90 = 25XP)
    XP_PER_DOLLAR = {"passenger": 0.2, "deploy": 0.265}
    _TOTAL = re.compile(r"^\W*\+?\W*[$S]\s*([0-9OoIlZB][0-9OoIlZB.,]{0,8})\W*$")

    def _read_centerfeed(self, frame, ev):
        """XP from the bottom-centre reward list (the boxed '+$3,500' running total and the newest
        ~4 entries under it, which scroll up; money and XP are separate entries).
        Only passengers / tactical deployments are counted here: XP = how much the running total
        grew x XP_PER_DOLLAR (the total reads reliably even when single lines don't). All other
        XP (building, revives ...) comes from the top-right feed (overlay: H.xp)."""
        lines = sorted(self.cf_lines.read(crop(frame, "centerfeed")), key=lambda x: x[2][0])
        total, entries = None, []
        for t, c, _ in lines:
            if c < 0.4:
                continue
            m = self._TOTAL.match(t.upper())
            if m and total is None:
                v = re.sub(r"[.,]", "", m.group(1)).translate(self._LOOK)
                total = int(v) if v.isdigit() else None
                continue
            e = self._feed_entry(t)
            if e:
                entries.append(e)
        if total is None and not entries:
            if self.t - self._cf_seen > 1.5:          # list gone: next one is a new burst
                self._cf, self._cf_raw, self._cf_total, self._cf_kind = [], None, 0, None
            return
        self._cf_seen = self.t
        kinds = [k for k, _, _ in entries if k in self.XP_PER_DOLLAR]
        if kinds:
            kind = max(set(kinds), key=kinds.count)
            # a different reward type twice in a row = a new list right after the last one
            if kind != self._cf_kind and kind == getattr(self, "_cf_kind_cand", None) and self._cf_kind:
                self._cf_total = 0
                self._cf_kind = kind
            elif self._cf_kind is None:
                self._cf_kind = kind
            self._cf_kind_cand = kind

        # 1) running total (needs the same value twice in a row)
        if total is not None:
            if total == self._cf_tcand and total != self._cf_total:
                if total < self._cf_total:            # a new list started without a gap
                    self._cf_total = 0
                grew = total - self._cf_total
                if self._cf_kind in self.XP_PER_DOLLAR:   # type unknown yet: keep the baseline, count it all later
                    if 0 < grew <= 20000:
                        ev.append(dict(type="xpgain", kind=self._cf_kind,
                                       amount=round(grew * self.XP_PER_DOLLAR[self._cf_kind])))
                    self._cf_total = total
            self._cf_tcand = total


    # most XP one line can give (seen: kill 250-350, headshot 275, shutdown 300, revive 250, heal 50,
    # passenger 100, deployment 50, supplies 500-600, building 1-10): a bigger number is a misread
    XP_MAX = dict(kill=800, headshot=800, shutdown=800, kill_assist=600, build=60, build_complete=60,
                  heal=300, revive=600, revive_assist=600, spot=300, spot_destroy=400, zone=300,
                  zone_enter=300, passenger=400, deploy=300, supplies=1500)

    def _read_xp(self, frame, ev):
        """Top-right reward feed: one event for each line's money and one for its XP (see feed.py)."""
        if self.feed_sx is None and self.tpl.stretch:
            self.feed.sx = self.tpl.stretch
        for kind, unit, amt, words, line in self.feed.update(self.feed.read(self._full), self.t):
            text = f"{words} {amt}XP" if unit == "xp" else f"{words} ${amt}"
            if any(i in words for i in IGNORE):
                continue
            if kind is None:
                ev.append(dict(type="line", region="xpfeed", text=text))   # (session log: new reward names)
                if unit != "xp" or amt > 300:
                    continue
                kind = "other"                      # unknown or unreadable words: still count the XP
            if unit == "xp" and amt > self.XP_MAX.get(kind, 2000):
                amt = amt % 1000 if amt % 1000 <= self.XP_MAX.get(kind, 2000) else 0   # "1250XP" = "|250XP"
                if not amt:
                    continue
            ev.append(dict(type="xp", kind=kind, amount=amt if unit == "xp" else 0,
                           money=amt if unit == "$" else 0, text=text, line=line))

    _POP_TOTAL = re.compile(r"^\W*\+?\W*[$S]\s*([0-9OoIlZB][0-9OoIlZB.,]{2,8})\W*$")

    def _read_popup(self, frame, ev):
        """Kill popup under the crosshair: a boxed running total ('+$4,000') over 'KILL CONFIRMED +$4,000',
        whose money turns into its XP after ~2 s. It stays ~5 s; a second kill in that time raises the
        total. One kill per new popup, plus one per rise of the total. (The kill's XP comes from the
        top-right feed's "KILL 250XP" line.)"""
        img = crop(frame, "popup")
        if int(stroke_mask(img).sum()) < 120:
            self._pop_gone()
            return
        lines = self.pop_lines.read(img)              # lines already read are reused (cockpit, scope text)
        txt = " ".join(norm(x[0]) for x in lines)
        # ("TONIGHT": our own overlay's kill card, only ever seen when testing on stream video)
        # a kill always says KILL CONFIRMED (a bare "$2,000" can be a price on the FOB supplies menu)
        if "CONFIRM" not in txt or "TONIGHT" in txt:
            self._pop_gone()
            return
        self._pop_seen = self.t
        total = None
        for t, c, _ in lines:
            m = self._POP_TOTAL.match(t.upper().replace(" ", ""))
            if m and c > 0.6:
                v = re.sub(r"[.,]", "", m.group(1)).translate(self._LOOK)
                total = int(v) if v.isdigit() else None
                break
        if self._pop is None:                       # a new popup: a kill
            self._pop = dict(total=None, cand=None, kills=1)
            self._kill_signal("popup", None, "")
        if total is None:
            return
        if total != self._pop["cand"]:              # every total must read the same twice
            self._pop["cand"] = total
            return
        if self._pop["total"] is None:
            self._pop["total"] = total
        elif total - self._pop["total"] >= 1400:    # it went up by another kill's worth ($1,500+)
            self._pop.update(total=total, kills=self._pop["kills"] + 1)
            self._kill_signal("popup", None, "")

    def _pop_gone(self):
        if self._pop is not None and self.t - self._pop_seen > 2.5:     # it fades / misreads for a moment
            self._pop = None

    def _read_killfeed(self, frame, ev):
        """Lobby kill feed lines with his name ("[KA]Benged [5m] victim" / "killer [11m] [KA]Benged").
        OCR reads the same line a little differently every time ("BananaBass", "BennaBass"), so a line
        counts the first time it reads cleanly (fast) and then is remembered by the other name, loosely,
        for as long as it stays on screen."""
        lines = self.kf_lines.read(crop(frame, "killfeed"))
        for raw, conf, _ in lines:
            if conf < 0.65:
                continue
            k = norm(raw)
            if not self._is_me(k):
                continue
            d = DIST_RE.search(raw)
            if not d:
                continue
            dist = int(d.group(1))
            left, right = raw[:d.start()], raw[d.end():]
            me_l, me_r = self._is_me(norm(left)), self._is_me(norm(right))
            other = norm(right if me_l else left)
            side = "self" if me_l and me_r else "kill" if me_l else "death"
            seen = None
            for r in self._kf_seen:
                if r["side"] == side and (r["other"] == other or difflib.SequenceMatcher(None, r["other"], other).ratio() >= 0.6):
                    seen = r
                    break
            if seen:
                seen["t"] = self.t
                continue
            self._kf_seen.append(dict(side=side, other=other, t=self.t))
            if side == "self":
                self._selfkill(ev, pilot=self.t - self._pilot_last < 12, via="killfeed", text=raw)
            elif side == "kill":
                self._kill_signal("feed", dist, right.strip(" []|"))
            else:
                ev.append(dict(type="killed_by", killer=left.strip(" []|"), dist=dist, text=raw))
        self._kf_seen = [r for r in self._kf_seen if self.t - r["t"] < 4.0]

    def _read_seats(self, frame, ev):
        """Who's in the vehicle: the list under "UNLOCKED [L]" in the control hints, in seat order
        ("[KA] Benged", "iamahumam", "TTV_QuantumLag", then the vehicle name "MH-6"). Shotgun = the
        name right after his. Sent when two reads agree."""
        lines = self.seat_lines.read(crop(frame, "hints"))
        names, after = [], False
        for raw, conf, _ in lines:
            n = norm(raw)
            if n.endswith("LOCKED") or n.endswith("LOCKEDL"):
                after = True
                continue
            if after and conf > 0.6 and len(n) >= 3:
                names.append(raw.strip(" @©®○●"))
        if len(names) >= 2 and re.fullmatch(r"[A-Z0-9\- ]{2,10}", names[-1]):
            names = names[:-1]                    # the vehicle name ("MH-6", "M1151")
        me = next((i for i, x in enumerate(names) if self._is_me(norm(x))), None)
        if me is None:
            return
        shot = names[me + 1] if me + 1 < len(names) else ""
        shot = re.sub(r"^[^A-Za-z0-9\[]+", "", shot)
        if shot == getattr(self, "_shot_cand", None) and shot != getattr(self, "_shot", None):
            self._shot = shot
            ev.append(dict(type="seats", shotgun=shot, names=names))
        self._shot_cand = shot

    # heli: ALT box sits this far right of the SPD box (template pixels, per look variant)
    ALT_DX = {0: 235, 1: 177, 2: 176}          # 2 = benged's fullscreen 1440x900 HUD

    def _find_gauges(self, gray, thresh=0.8):
        """Heli SPD + ALT boxes (pilot seat only). Ground vehicles have an SPD box but no ALT box."""
        # ALT is the stronger match; then SPD must sit exactly where it belongs, left of ALT on the
        # same row. That position check is strict, so SPD itself may score a little lower.
        al = self.tpl.locate(gray, "alt_label", stop=0.9)
        if al[0] < thresh:
            return None
        xs = al[5] / al[3]
        sp = self.tpl.locate_near(gray, "spd_label", al[1] - self.ALT_DX[al[4]] * al[5], al[2],
                                  al[3], al[4], round(xs, 3) if round(xs, 3) in (1.111, 1.333) else 1.0)
        if sp[0] < thresh - 0.08:
            return None
        if abs((al[1] - sp[1]) - self.ALT_DX[sp[4]] * sp[5]) > 14 or abs(al[2] - sp[2]) > 5:
            return None
        return sp, al

    def _find_vehicle_gauge(self, gray, thresh=0.8):
        v = self.tpl.locate(gray, "vspd_label", stop=0.9)
        return v if v[0] >= thresh else None

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
                  ("alt", 0): (14, 29, 66), ("alt", 1): (14, 29, 54),
                  ("spd", 2): (14, 29, 55), ("alt", 2): (14, 29, 54)}

    def _gauge_text(self, frame, key, match):
        _, x, y, s, v, sx = match
        top, bot, right = self.GAUGE_LINE[(key, v)]
        return read_line(frame[int(y + top * s):int(y + bot * s), max(0, x - 2):int(x + right * sx)])[0]

    def _read_vehicle_speed(self, frame, ev):
        """Ground vehicles: '79  KM/H' under the SPD box (number first)."""
        _, x, y, s, _, sx = self._vgauge
        txt = read_line(frame[int(y + 14 * s):int(y + 29 * s), max(0, x - 2):int(x + 56 * sx)])[0]
        m = re.match(r"\s*([\dOoIlSB]{1,3})", txt)
        n = _digits(m.group(1)) if m else ""
        spd = int(n) if n.isdigit() else None
        spd = self._steady("spd", spd if spd is not None and spd <= 200 else None, 40)
        if spd is not None:
            ev.append(dict(type="speed", spd=spd, raw=txt))

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
    # A kill shows as the popup under the crosshair AND a kill-feed line, up to a few seconds apart.
    # The first one to show is sent at once (fast); the other one, when it comes, only adds what
    # the first lacked (distance, victim) as a kill_info event instead of counting a second kill.
    def _kill_signal(self, via, dist, victim):
        for k in self._kills:
            if via not in k["via"] and self.t - k["t0"] < 8.0:
                k["via"].append(via)
                if (dist and not k["dist"]) or (victim and not k["victim"]):
                    k["dist"], k["victim"] = k["dist"] or dist, k["victim"] or victim
                    self._kills_ready.append(dict(type="kill_info", id=k["id"], dist=k["dist"], victim=k["victim"]))
                return
        self._kill_n = getattr(self, "_kill_n", 0) + 1
        k = dict(id=self._kill_n, t0=self.t, via=[via], dist=dist, victim=victim)
        self._kills = [x for x in self._kills if self.t - x["t0"] < 8.0] + [k]
        self._kills_ready.append(dict(type="kill", id=k["id"], dist=dist, victim=victim, via=via))

    def _flush_kill(self, ev):
        ev.extend(self._kills_ready)
        self._kills_ready = []

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
