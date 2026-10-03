"""Self-update from GitHub (public repo, no login needed).

The repo holds exactly the files in benged's app folder. Each release is a git tag
vX.Y.Z; the VERSION file on main says which tag is newest (first line = version,
the rest = what changed). Updating:
  * downloads that tag as a zip
  * backs up every file it replaces into .backup/<old version>/ (rollback)
  * never touches config.json, .venv, .backup, or the .bat files (a running .bat
    must not be rewritten; they only get written if missing)
  * reinstalls packages only when requirements.txt changed
"""
import io
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile

REPO = "kevinwsmg-bit/benged-network"
HERE = os.path.dirname(os.path.abspath(__file__))
KEEP = {"config.json", ".venv", ".backup", "debug", "__pycache__", ".git"}
UA = {"User-Agent": "benged-network-updater"}


def _read(path, default=""):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return default


def current():
    return (_read(os.path.join(HERE, "VERSION"), "0").strip().splitlines() or ["0"])[0].strip()


def vt(v):
    return tuple(int(x) for x in re.findall(r"\d+", v)) or (0,)


def latest(timeout=6):
    """(version, notes) of the newest release on GitHub."""
    req = urllib.request.Request(f"https://api.github.com/repos/{REPO}/contents/VERSION?ref=main",
                                 headers=dict(UA, Accept="application/vnd.github.raw"))
    text = urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8").strip()
    lines = text.splitlines()
    return lines[0].strip(), "\n".join(lines[1:]).strip()


def newer(timeout=6):
    """(version, notes) if GitHub has something newer than what's installed, else None."""
    v, notes = latest(timeout)
    return (v, notes) if vt(v) > vt(current()) else None


def apply(version, log=print):
    old = current()
    req = urllib.request.Request(f"https://codeload.github.com/{REPO}/zip/refs/tags/v{version}", headers=UA)
    z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(req, timeout=90).read()))
    root = z.namelist()[0].split("/")[0] + "/"
    backup = os.path.join(HERE, ".backup", old)
    old_req = _read(os.path.join(HERE, "requirements.txt"))
    changed = 0
    for info in z.infolist():
        rel = info.filename[len(root):]
        if not rel or rel.endswith("/") or rel.split("/")[0] in KEEP:
            continue
        dst = os.path.join(HERE, *rel.split("/"))
        if rel.lower().endswith(".bat") and os.path.exists(dst):
            continue
        data = z.read(info)
        if os.path.exists(dst):
            with open(dst, "rb") as f:
                if f.read() == data:
                    continue
            bk = os.path.join(backup, *rel.split("/"))
            os.makedirs(os.path.dirname(bk), exist_ok=True)
            shutil.copy2(dst, bk)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(data)
        changed += 1
    with open(os.path.join(HERE, ".backup", "last.json"), "w", encoding="utf-8") as f:
        json.dump(dict(previous=old, installed=version), f)
    log(f"  {changed} file(s) updated.")
    if _read(os.path.join(HERE, "requirements.txt")) != old_req:
        log("  Installing new packages (a few minutes) ...")
        subprocess.call([sys.executable, "-m", "pip", "install", "-r", os.path.join(HERE, "requirements.txt")])


def rollback(log=print):
    """Put back the files the last update replaced."""
    try:
        with open(os.path.join(HERE, ".backup", "last.json"), encoding="utf-8") as f:
            prev = json.load(f)["previous"]
    except (OSError, ValueError, KeyError):
        log("  Nothing to roll back.")
        return False
    src = os.path.join(HERE, ".backup", prev)
    for dirpath, _, files in os.walk(src):
        for fn in files:
            s = os.path.join(dirpath, fn)
            d = os.path.join(HERE, os.path.relpath(s, src))
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copy2(s, d)
    log(f"  Rolled back to {prev}.")
    return True


if __name__ == "__main__":
    if "--rollback" in sys.argv:
        rollback()
    else:
        n = newer()
        print(f"  Installed {current()}, newest {n[0] if n else current()}")
