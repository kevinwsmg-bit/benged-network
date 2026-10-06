"""The top-right reward feed ("BUILDING 1XP", "TARGET SPOTTED +$250", ...).

How the game shows it (measured on benged's 2026-10-05 stream):
  * every line is ONE reward. It first shows the money ("BUILDING +$8"); about two seconds later
    the same line turns into its XP ("BUILDING 1XP") and gets its role icon (wrench, crosshair ...)
  * lines sit on a fixed grid of 6 slots under the money boxes; a new line is added at the
    bottom, the top line leaves after ~10 s (or when a 7th line arrives) and the rest move up
  * the same reward can repeat many times in a row (building: "BUILDING 1XP" x6)

So lines are followed by slot, not by text. Each pass reads every slot's amount ("1XP", "+$8",
a fixed column) and, once per line, its words; then lines the new read up with the list we
already know (how many left at the top, how many are new at the bottom). Identical lines are
told apart by their age and by the money -> XP turn (a line never turns back).
Each line reports its money once and its XP once.

OCR is the cost (~18 ms a call), so a slot whose picture hasn't changed since the last pass
(or has just moved up one slot) reuses what was read there.
"""
import re
import difflib
from collections import Counter

import cv2
import numpy as np

from .ocr import engine

# geometry in game pixels at 1080 tall, measured from the right edge (scaled by height, and by
# the sideways stretch when a 16:10 game is stretched to 16:9)
TOP = 0.045                       # feed region top (fraction of the height)
BASE, PITCH, SLOTS = 18.8, 21.2, 6
HALF = 8                          # half height of a line band
AMT = (100, 31)                   # amount column ("1XP", "250XP", "+$1,500"; new money sits in a white box)
ICON = (31, 16)                   # role icon (XP lines only)
WORDS = (300, 68)                 # words, right-aligned before the amount
LIFE = 9.5                        # seconds a line stays (sooner when the list is full)
MAX_AMT, MAX_WORDS = 5, 2         # OCR calls per pass (the rest wait for the next pass)

_LOOK = str.maketrans("OoQDIl|SZB", "0000111528")
_XP = re.compile(r"([0-9OoQDIl|SZB]{1,4})\s*X\s*[PR]")
_MONEY = re.compile(r"(-?)\s*\+?\s*[$S8]\s*([0-9OoIlZB][0-9OoIlZB.,]{0,7})")


def parse_amount(raw):
    """'10XP' -> ('xp', 10); '+S35' -> ('$', 35); '-$500' -> ('$', -500); else None."""
    u = raw.upper().replace(" ", "")
    m = _XP.search(u)
    if m:
        v = m.group(1).translate(_LOOK)
        return ("xp", int(v)) if v.isdigit() and int(v) > 0 else None
    m = _MONEY.search(u)
    if m:
        v = re.sub(r"[.,]", "", m.group(2)).translate(_LOOK)
        if v.isdigit() and int(v) > 0:
            return "$", -int(v) if m.group(1) == "-" else int(v)
    return None


def clean_words(raw):
    return re.sub(r"[^A-Z]", "", raw.upper())


def same_words(a, b):
    if not a or not b or a == b:
        return True
    n = min(len(a), len(b))
    # OCR adds/drops letters at the start of a line ("MSBUILDING"): compare the ends too
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.75 or \
        (n >= 6 and difflib.SequenceMatcher(None, a[-n:], b[-n:]).ratio() >= 0.85)


class Line:
    __slots__ = ("words", "reads", "born", "seen", "sent", "id")
    n = 0

    def __init__(self, t):
        Line.n += 1
        self.id = Line.n            # the overlay counts a reward once per line (money and XP are 2 events)
        self.born = self.seen = t
        self.words = Counter()
        self.reads = {"$": Counter(), "xp": Counter()}
        self.sent = set()

    @property
    def word(self):
        return self.words.most_common(1)[0][0] if self.words else ""


class Slot:
    """What one pass saw in one slot."""
    __slots__ = ("sig", "asig", "amount", "words", "icon", "red", "read")

    def __init__(self, sig, asig, icon, red=False):
        self.sig, self.asig, self.icon, self.red = sig, asig, icon, red
        self.amount = None          # ('xp', 1) / ('$', 8) / None = unread or unreadable
        self.words = ""
        self.read = False           # amount OCR done (or copied)


class Feed:
    # rewards that come in fast streams (a line can leave before its XP shows): estimate those
    ESTIMATE = ("build", "build_complete")

    def __init__(self, classify):
        self.classify = classify
        self.lines = []
        self.prev = {}              # slot -> Slot of the last pass
        self.ocr_calls = 0
        self.sx = 1.0               # sideways stretch of the picture
        self.rate = {}              # kind -> recent XP-per-$ of lines seen both ways

    # ------------------------------------------------------------ reading
    def _cols(self, W, s, a, b):
        return W - int(a * s * self.sx), W - int(b * s * self.sx)

    def read(self, full):
        H, W = full.shape[:2]
        s = H / 1080
        y0 = TOP * H
        ax0, ax1 = self._cols(W, s, *AMT)
        ix0, ix1 = self._cols(W, s, *ICON)
        wx0, wx1 = self._cols(W, s, *WORDS)
        out, bands = {}, {}
        k9 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        for k in range(SLOTS):
            c = y0 + (BASE + k * PITCH) * s
            ya, yb = int(c - HALF * s), int(c + HALF * s)
            amt = full[ya:yb, ax0:ax1]
            mn = amt.min(axis=2)
            th = cv2.morphologyEx(mn, cv2.MORPH_TOPHAT, k9)
            # something written in the amount column: bright thin strokes, or the white box
            box = float((mn > 200).mean()) > 0.45
            if not box and int(((mn > 150) & (th > 30)).sum()) < 10 * s * s:
                continue
            ib = full[ya:yb, ix0:ix1]
            icon = ib.min(axis=2)
            b, g, r = (ib[:, :, i].astype(np.int16) for i in range(3))
            red = int(((r > 150) & (g < 90) & (b < 90)).sum()) >= 6 * s * s   # red marker = money going out (tips sent)
            sig = cv2.resize(cv2.cvtColor(full[ya:yb, wx0:ix1], cv2.COLOR_BGR2GRAY), (64, 8),
                             interpolation=cv2.INTER_AREA).astype(np.int16)
            # the amount column on its own, finer: lines that differ only in their amount ("250XP" / "49XP")
            asig = cv2.resize(cv2.cvtColor(amt, cv2.COLOR_BGR2GRAY), (32, 10), interpolation=cv2.INTER_AREA).astype(np.int16)
            out[k] = Slot(sig, asig, int((icon > 170).sum()) >= 20 * s * s, red)
            bands[k] = (amt, full[ya:yb, wx0:wx1])
        # same picture as in the last pass in the same slot, or the one below (moved up): reuse what was read
        for k, sl in out.items():
            for j in (k, k + 1):
                p = self.prev.get(j)
                if p and p.amount and float(np.abs(sl.sig - p.sig).mean()) < 6                         and float(np.abs(sl.asig - p.asig).mean()) < 5:
                    sl.amount, sl.words, sl.read = p.amount, p.words, True
                    break
        n = 0
        self._turn = getattr(self, "_turn", 0) + 1
        order = sorted(out, reverse=True)                 # newest (bottom) first, but take turns
        order = order[self._turn % 2::2] + order[1 - self._turn % 2::2] if self._turn % 3 == 0 else order
        for k in order:
            sl = out[k]
            if sl.read or n >= MAX_AMT:
                continue
            img = cv2.resize(bands[k][0], None, fx=2 / s, fy=2 / s, interpolation=cv2.INTER_CUBIC)
            res, _ = engine()(img, use_det=False, use_cls=False)
            self.ocr_calls += 1
            n += 1
            sl.read = True
            if res and res[0][1] >= 0.6:
                sl.amount = parse_amount(res[0][0])
        self._bands = bands
        self.prev = out
        return out

    def _read_words(self, k, sl):
        res, _ = engine()(self._bands[k][1], use_det=False, use_cls=False)
        self.ocr_calls += 1
        sl.words = clean_words(res[0][0]) if res and res[0][1] >= 0.6 else ""

    # ----------------------------------------------------------- tracking
    @staticmethod
    def _score(line, sl):
        if sl is None:
            return -1.0                              # a known line should be here
        sc = 0.0
        if sl.words and line.words:
            sc += 1.0 if same_words(line.word, sl.words) else -3.0
        if sl.amount is None:
            return sc
        unit, amt = sl.amount
        if unit == "$" and line.reads["xp"]:
            return sc - 2.5                          # a line never turns back from XP to money
        seen = line.reads[unit]
        if seen:
            sc += 1.5 if (amt in seen or -amt in seen) else -1.0
        elif unit == "xp":
            sc += 0.3                                # money -> XP turn
        if sl.icon and unit == "$":
            sc -= 0.5
        return sc

    def update(self, slots, t):
        """slots from read(); returns finished events [(kind, unit, amount, words, line id)]."""
        L = self.lines
        events = []
        if not slots:
            # nothing in the amount column at all: the list is empty (or unreadable for a moment)
            while L and (t - L[0].seen > 1.5 or t - L[0].born > LIFE):
                events += self._flush(L.pop(0))
            return events
        n = max(slots) + 1
        cur = [slots.get(k) for k in range(n)]
        expect = 0
        while expect < len(L) and t - L[expect].born > LIFE:
            expect += 1
        best = None
        for d in range(len(L) + 1):
            rem = L[d:]
            sc = sum(self._score(rem[k], cur[k]) for k in range(min(len(rem), n)))
            sc -= 2.0 * max(0, len(rem) - n)          # known lines can't vanish from the bottom
            new = sum(1 for k in range(len(rem), n) if cur[k] is not None)
            sc -= 3.0 * max(0, len(rem) + new - SLOTS)
            sc -= 0.8 * abs(d - expect)               # lines leave ~LIFE s after they came
            key = (round(sc, 3), -abs(d - expect), -d)
            if best is None or key > best[0]:
                best = (key, d)
        d = best[1]
        for ln in L[:d]:
            events += self._flush(ln)
        L = L[d:]
        for k, sl in enumerate(cur):
            if sl is None:
                continue
            while k >= len(L):
                L.append(Line(t))
            ln = L[k]
            ln.seen = t
            if sl.amount:
                unit, amt = sl.amount
                if unit == "$" and sl.red:
                    amt = -abs(amt)
                ln.reads[unit][amt] += 1
            if sl.words:
                ln.words[sl.words] += 1
        # words: once per line (newest first), a couple per pass
        budget = MAX_WORDS
        for k in range(min(len(L), n) - 1, -1, -1):
            sl = cur[k]
            if budget and sl is not None and not sl.words and not L[k].words:
                self._read_words(k, sl)
                budget -= 1
                if sl.words:
                    L[k].words[sl.words] += 1
        for ln in L:
            events += self._ready(ln)
        self.lines = L
        return events

    def _kind(self, ln):
        for w, _ in ln.words.most_common():
            k = self.classify(w)
            if k:
                return k, w
        # badly read words ("UDING"): the closest known reward
        from .reader import XP_RULES
        best = max(((difflib.SequenceMatcher(None, w, p).ratio(), v, w) for w in ln.words for p, v in XP_RULES
                    if len(p) >= 5), default=(0, None, ""))
        if best[0] >= 0.7:
            return best[1], best[2]
        return None, ln.word

    def _ready(self, ln, final=False):
        out = []
        if not ln.words and not final:
            return out
        kind, w = self._kind(ln)
        for unit in ("$", "xp"):
            if unit in ln.sent or not ln.reads[unit]:
                continue
            amt, c = ln.reads[unit].most_common(1)[0]
            if c >= 2 or final:
                ln.sent.add(unit)
                out.append((kind, unit, amt, w, ln.id))
        if "$" in ln.sent and "xp" in ln.sent and kind and "rate" not in ln.sent:
            ln.sent.add("rate")
            m, x = ln.reads["$"].most_common(1)[0][0], ln.reads["xp"].most_common(1)[0][0]
            if m > 0:
                self.rate.setdefault(kind, []).append(x / m)
                self.rate[kind] = self.rate[kind][-15:]
        if final and "xp" not in ln.sent and ln.reads["$"] and kind in self.rate and kind in self.ESTIMATE:
            # left before its XP was read (busy list): estimate the XP from its money
            m = ln.reads["$"].most_common(1)[0][0]
            r = min(0.4, sorted(self.rate[kind])[len(self.rate[kind]) // 2])
            if m > 0 and round(m * r) > 0:
                out.append((kind, "xp", round(m * r), w + " (est)", ln.id))
        return out

    def _flush(self, ln):
        """A line leaving: report anything it showed only once."""
        return self._ready(ln, final=True)
