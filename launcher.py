"""What START.bat runs: update if there's something new, then run the server.

  * on start: installs a newer version from GitHub if there is one (offline = skip)
  * the control panel's "Update now" makes the server exit with code 3 -> update + restart
  * if a fresh update crashes within 30 s, the previous version is put back automatically
"""
import os
import subprocess
import sys
import time

import updater

HERE = os.path.dirname(os.path.abspath(__file__))
UPDATE_REQUESTED = 3


def try_update():
    try:
        n = updater.newer()
    except Exception as e:
        print(f"  (Couldn't check for updates: {e}. Starting the version you have.)")
        return False
    if not n:
        print(f"  Benged Network {updater.current()} is up to date.")
        return False
    print(f"  Updating Benged Network {updater.current()} -> {n[0]} ...")
    try:
        updater.apply(n[0])
    except Exception as e:
        print(f"  Update failed ({e}). Starting the version you have.")
        return False
    print("  Updated.")
    return True


def main():
    just_updated = False if os.environ.get("BN_NO_STARTUP_UPDATE") else try_update()   # env var: testing only
    while True:
        t0 = time.time()
        code = subprocess.call([sys.executable, os.path.join(HERE, "server.py")] + sys.argv[1:])
        if code == UPDATE_REQUESTED:
            just_updated = try_update()
            continue
        if code != 0 and just_updated and time.time() - t0 < 30:
            print("\n  The new version crashed while starting. Going back to the previous one.\n")
            updater.rollback()
            just_updated = False
            continue
        return code


if __name__ == "__main__":
    sys.exit(main())
