"""Switch on Rocket League's Stats API (the official setting; nothing else in the game is touched).

The setting lives in <Rocket League>\\TAGame\\Config\\TAStatsAPI.ini (or DefaultStatsAPI.ini when that
doesn't exist), section [TAGame.MatchStatsExporter_TA]. Rocket League must be restarted afterwards.
"""
import glob
import json
import os
import re

SECTION = "[TAGame.MatchStatsExporter_TA]"
WANT = {"PacketSendRate": "10", "Port": "49123", "WebPort": "49124"}   # 10 state updates/s is plenty; events are instant anyway


def find_installs():
    """Rocket League folders on this PC (Epic and Steam, any drive)."""
    found = []
    # Epic: the launcher's manifests know the install folder ("Sugar" is Rocket League's codename)
    for m in glob.glob(r"C:\ProgramData\Epic\EpicGamesLauncher\Data\Manifests\*.item"):
        try:
            d = json.load(open(m, encoding="utf-8"))
            if "rocket" in (d.get("DisplayName", "") + d.get("AppName", "")).lower() or d.get("AppName") == "Sugar":
                found.append(d.get("InstallLocation", ""))
        except (OSError, ValueError):
            pass
    # Steam: every library folder
    libs = [r"C:\Program Files (x86)\Steam"]
    for vdf in (r"C:\Program Files (x86)\Steam\steamapps\libraryfolders.vdf",):
        try:
            libs += re.findall(r'"path"\s+"([^"]+)"', open(vdf, encoding="utf-8").read())
        except OSError:
            pass
    for lib in libs:
        found.append(os.path.join(lib.replace("\\\\", "\\"), "steamapps", "common", "rocketleague"))
    for drive in "CDEFG":
        found += [rf"{drive}:\Program Files\Epic Games\rocketleague", rf"{drive}:\Epic Games\rocketleague"]
    out = []
    for f in found:
        if f and os.path.isdir(os.path.join(f, "TAGame", "Config")) and f not in out:
            out.append(f)
    return out


def ini_path(install):
    cfg = os.path.join(install, "TAGame", "Config")
    ta = os.path.join(cfg, "TAStatsAPI.ini")
    return ta if os.path.exists(ta) else os.path.join(cfg, "DefaultStatsAPI.ini")


def status(install):
    """{'path', 'enabled', 'settings'} for one install."""
    p = ini_path(install)
    vals = {}
    try:
        sect = None
        for line in open(p, encoding="utf-8", errors="ignore"):
            s = line.strip()
            if s.startswith("["):
                sect = s
            elif sect == SECTION and "=" in s:
                k, v = s.split("=", 1)
                vals[k.strip()] = v.strip()
    except OSError:
        pass
    try:
        rate = float(vals.get("PacketSendRate", "0"))
    except ValueError:
        rate = 0
    return dict(path=p, enabled=rate > 0, settings=vals)


def enable(install):
    """Write the settings (keeps anything else in the file). Returns status(); raises PermissionError
    when Windows won't let us write there (then run START.bat as administrator once, or edit by hand)."""
    p = ini_path(install)
    try:
        lines = open(p, encoding="utf-8", errors="ignore").read().splitlines()
    except OSError:
        lines = []
    out, in_sect, seen, done = [], False, set(), False
    for line in lines:
        s = line.strip()
        if s.startswith("["):
            if in_sect and not done:
                gap = []
                while out and not out[-1].strip():      # keep the blank line(s) before the next section
                    gap.append(out.pop())
                out += [f"{k}={v}" for k, v in WANT.items() if k not in seen] + gap
                done = True
            in_sect = s == SECTION
            out.append(line)
            continue
        if in_sect and "=" in s and s.split("=", 1)[0].strip() in WANT:
            k = s.split("=", 1)[0].strip()
            out.append(f"{k}={WANT[k]}")
            seen.add(k)
            continue
        out.append(line)
    if in_sect and not done:
        out += [f"{k}={v}" for k, v in WANT.items() if k not in seen]
    elif SECTION not in [l.strip() for l in lines]:
        out += ["", SECTION] + [f"{k}={v}" for k, v in WANT.items()]
    with open(p, "w", encoding="utf-8", newline="\r\n") as f:
        f.write("\n".join(out) + "\n")
    return status(install)
