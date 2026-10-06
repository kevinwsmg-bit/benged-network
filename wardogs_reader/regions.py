"""Screen regions for the WARDOGS HUD, as fractions of the frame (x0, y0, x1, y1).

Measured on benged's 16:9 streams. The reader always asks OBS for a 1280x720
screenshot, so pixel sizes in the detectors assume a 720p frame.
"""

REGIONS = {
    # top-right: match cash + persistent wallet, one line
    "cash": (0.76, 0.0, 1.0, 0.048),
    # top-right under the cash: XP / money event feed, right-aligned lines
    "xpfeed": (0.66, 0.045, 1.0, 0.24),
    # bottom centre, above the heli gauges: the reward list ("PASSENGER SURVIVED +$500",
    # "... 100XP"). Money and XP are separate entries; the newest 4 show and scroll up.
    "centerfeed": (0.36, 0.70, 0.64, 0.90),
    # left side, mid-screen: lobby kill feed ("KILLER [weapon] [NN m] VICTIM")
    "killfeed": (0.0, 0.27, 0.36, 0.64),
    # centre, just under the crosshair: "+$2,000 CONFIRMED" kill popup
    "popup": (0.40, 0.55, 0.60, 0.76),
    # right edge: vehicle / heli control hints (COLLECTIVE LIFT = pilot seat)
    "hints": (0.78, 0.38, 1.0, 0.78),
    # right edge: the vehicle's seat list ("UNLOCKED [L]", then who sits where, then the vehicle name).
    # It hangs at the bottom of the control hints, so it sits lower in vehicles with fewer controls
    "seats": (0.78, 0.36, 1.0, 0.93),
    # bottom-right: equipped item name ("LARGE HAMMER // BUILD SUPPLIES" = building)
    "weapon": (0.78, 0.80, 1.0, 0.89),
    # vendor screens: the green Purchase button at the bottom right shows the cart total ("$350")
    "cart": (0.84, 0.87, 1.0, 0.985),
    # vendor "REBUY LAST LOADOUT" dialog (covers the vendor tabs)
    "rebuy": (0.15, 0.15, 0.55, 0.40),
    # bottom centre with a medkit in hand: "REVIVE FRIENDLY [lmb] (+) [rmb] HEAL SELF"
    "medkit": (0.25, 0.80, 0.75, 0.92),
    # bottom strip: GIVE UP / CALL FOR HELP (downed), map controls (dead / redeploy)
    "bottom": (0.08, 0.88, 0.92, 1.0),
    # right-bottom: VIEW DAMAGE LOG + who hit him
    "damagelog": (0.70, 0.55, 1.0, 0.92),
    # top strip: vendor / inventory tabs (EQUIPMENT VENDOR, INVENTORY, OVERVIEW ...)
    "top": (0.0, 0.0, 0.75, 0.05),
    # heli instrument boxes, bottom centre: "SPD / KM/H 277" and "ALT / AGL 56".
    # Cockpit view and third-person view put them in different places; the reader
    # finds the SPD / ALT labels in here and reads the number line under each.
    "gauges": (0.30, 0.78, 0.72, 0.93),
    # bottom-left: loading screen map name + "OFFICIAL WARDOGS | SERVER"
    "loading": (0.0, 0.70, 0.60, 0.92),
    # top-centre compass strip (heading)
    "compass": (0.36, 0.0, 0.64, 0.04),
}


def crop(frame, name):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = REGIONS[name]
    return frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
