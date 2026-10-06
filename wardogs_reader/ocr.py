"""Fast line reading for fixed HUD regions.

RapidOCR's text *detector* is slow on CPU (~0.5-1 s per crop), but its
*recogniser* reads one line in ~15 ms. HUD text sits in known regions, so we
find the lines ourselves with a morphological top-hat (thin bright strokes) and
a row projection, then run recognition only.
"""
import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

_ocr = None


def engine():
    global _ocr
    if _ocr is None:
        # 2 threads: fast enough (~20 ms a line) and never fights the game for CPU
        _ocr = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)
    return _ocr


_boxes = None


def read_boxes(img):
    """Find the separate text boxes in a small image and read each (left to right):
    [(x, text, conf)]. ~100 ms on the money strip. The 'max' limit stops RapidOCR from
    blowing a small strip up to 736 px first (that made it take ~1 s)."""
    global _boxes
    if _boxes is None:
        _boxes = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1,
                          det_limit_side_len=736, det_limit_type="max")
    res, _ = _boxes(img, use_det=True, use_cls=False, use_rec=True)
    return sorted((min(p[0] for p in b), t, float(c)) for b, t, c in (res or []))


def stroke_mask(img, thresh=38):
    """Bright thin strokes (HUD text and icons), insensitive to smooth bright sky."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k)
    return (th > thresh).astype(np.uint8)


def _split(rows, y, y2, split):
    """Cut a band taller than `split` rows (two lines touching) at its emptiest row, recursively."""
    if split is None or y2 - y <= split:
        return [(y, y2)]
    mid = y + 5 + int(np.argmin(rows[y + 5:y2 - 5]))
    return _split(rows, y, mid, split) + _split(rows, mid + 1, y2, split)


def find_lines(img, min_h=6, max_h=30, min_px=6, pad=3, split=None):
    """Return (y0, y1, x0, x1) boxes for text lines inside a region crop.
    split: cut bands taller than this many rows (lines packed close together, e.g. seat lists)."""
    m = stroke_mask(img)
    rows = m.sum(axis=1)
    on = rows >= min_px
    boxes, y = [], 0
    H = len(rows)
    while y < H:
        if on[y]:
            y2 = y
            while y2 < H and on[y2]:
                y2 += 1
            for (ya, yb) in _split(rows, y, y2, split):
                if min_h <= (yb - ya) <= max_h:
                    band = m[ya:yb]
                    cols = np.where(band.sum(axis=0) > 0)[0]
                    if len(cols):
                        boxes.append((max(0, ya - pad), min(H, yb + pad),
                                      max(0, cols[0] - pad), min(img.shape[1], cols[-1] + pad)))
            y = y2
            continue
            if min_h <= (y2 - y) <= max_h:
                band = m[y:y2]
                cols = np.where(band.sum(axis=0) > 0)[0]
                if len(cols):
                    boxes.append((max(0, y - pad), min(H, y2 + pad),
                                  max(0, cols[0] - pad), min(img.shape[1], cols[-1] + pad)))
            y = y2
        else:
            y += 1
    return boxes


def read_line(img):
    """Recognise a single text line. Returns (text, confidence)."""
    if img.size == 0 or img.shape[0] < 4 or img.shape[1] < 8:
        return "", 0.0
    up = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    res, _ = engine()(up, use_det=False, use_cls=False)
    if not res:
        return "", 0.0
    text, conf = res[0][0], float(res[0][1])
    return text, conf


def read_lines(img, **kw):
    """Find and read every line in a region. Returns list of (text, conf, box)."""
    out = []
    for b in find_lines(img, **kw):
        y0, y1, x0, x1 = b
        t, c = read_line(img[y0:y1, x0:x1])
        if t:
            out.append((t, c, b))
    return out


class CachedLines:
    """Reads feed lines but only runs OCR on lines it has not seen before.

    Each candidate line is fingerprinted by its stroke mask (the text itself,
    mostly independent of the scenery behind it). Known fingerprints reuse the
    earlier result, so a line that sits on screen for 6 seconds is read once.
    Bands that do not look like text (too sparse, too dense, too wide) are skipped.
    """

    def __init__(self, max_w=None, min_density=0.06, max_density=0.45, max_new=4, split=None):
        self.split = split
        self.cache = {}
        self.max_w = max_w
        self.min_d, self.max_d = min_density, max_density
        self.max_new = max_new
        self.ocr_calls = 0

    @staticmethod
    def _key(mask):
        small = cv2.resize(mask * 255, (40, 6), interpolation=cv2.INTER_AREA)
        return (small > 90).tobytes()

    def read(self, img):
        out, new = [], 0
        m_all = stroke_mask(img)
        for (y0, y1, x0, x1) in find_lines(img, split=self.split):
            if self.max_w and (x1 - x0) > self.max_w:
                continue
            m = m_all[y0:y1, x0:x1]
            d = float(m.mean())
            if not (self.min_d <= d <= self.max_d):
                continue
            k = self._key(m)
            if k in self.cache:
                t, c = self.cache[k]
            elif new < self.max_new:
                t, c = read_line(img[y0:y1, x0:x1])
                self.ocr_calls += 1
                new += 1
                self.cache[k] = (t, c)
                if len(self.cache) > 400:
                    self.cache.pop(next(iter(self.cache)))
            else:
                continue
            if t:
                out.append((t, c, (y0, y1, x0, x1)))
        return out


def norm(s):
    """Uppercase, letters/digits only. 'CONTROL ZONE PRESENCE +$160' -> 'CONTROLZONEPRESENCE160'."""
    return "".join(ch for ch in s.upper() if ch.isalnum())
