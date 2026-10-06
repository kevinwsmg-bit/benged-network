"""Benged Network overlay server.

Runs on benged's streaming PC next to OBS:
  * connects to OBS (WebSocket, Tools > WebSocket Server Settings)
  * screenshots his game capture source ~4x a second (never touches the game)
  * runs the WARDOGS reader on each screenshot
  * pushes events to the overlay (OBS browser source) and the control panel

    python server.py            ->  control panel: http://localhost:8765/
                                    overlay:       http://localhost:8765/overlay
"""
import argparse
import asyncio
import base64
import glob
import json
import os
import sys
import threading
import time
import traceback
import uuid

import cv2
import numpy as np
from aiohttp import web, WSMsgType

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wardogs_reader.reader import Reader          # noqa: E402
from wardogs_reader.regions import REGIONS        # noqa: E402
from wardogs_reader.ocr import norm               # noqa: E402
import updater                                    # noqa: E402

CONFIG_PATH = os.path.join(HERE, "config.json")
DEFAULTS = {
    "obs_host": "localhost",
    "obs_port": 4455,
    "obs_password": "",
    "source": "",              # name of the WARDOGS game capture source in OBS
    "names": ["Benged"],       # in-game names to treat as him (add alt accounts here)
    "fps": 3,
    "port": 8765,
    "replay_buffer": False,    # save OBS replay buffer on big moments (crash, long shot, 3+ streak)
}


def load_config():
    cfg = dict(DEFAULTS)
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def friendly(e):
    """Plain-English OBS errors for the control panel."""
    msg = f"{type(e).__name__}: {e}"
    low = msg.lower()
    if "refused" in low or "10061" in low or "timed out" in low or "timeout" in low:
        return "Can't reach OBS. Is OBS open, and is Tools > WebSocket Server Settings > Enable WebSocket server ticked (port 4455)?"
    if "auth" in low or "password" in low or "4009" in low:
        return "OBS rejected the password. In OBS press Show Connect Info, copy the password again, paste it in Step 1 and press Connect."
    if "no source was found" in low or "600" in low or "not found" in low:
        return "OBS can't find that source name. Press Refresh and pick the WARDOGS source again."
    return msg


class Hub:
    """Fan-out of events to every connected page."""

    def __init__(self):
        self.clients = set()
        self.overlays = set()                 # the subset that said hello as the OBS overlay
        self.loop = None
        self.log = []
        self.session = uuid.uuid4().hex[:8]   # new every START.bat: overlays reload themselves on a new id

    async def send(self, ev):
        self.log = (self.log + [ev])[-200:]
        dead = []
        for ws in self.clients:
            try:
                await ws.send_str(json.dumps(ev))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)
            self.overlays.discard(ws)

    def send_threadsafe(self, ev):
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.send(ev), self.loop)


class Capture(threading.Thread):
    """OBS screenshot -> reader -> events. Reconnects on its own."""

    def __init__(self, hub, cfg, replay=None):
        super().__init__(daemon=True)
        self.hub, self.cfg = hub, cfg
        self.replay = sorted(glob.glob(os.path.join(replay, "*.jpg"))) if replay else None
        self.replay_i = 0
        self.reader = Reader(names=cfg["names"])
        self.status = dict(obs="connecting", source=cfg["source"], fps=0.0, ms=0, frames=0, error="", paused=False)
        self.last_jpg = None
        self.cl = None
        self.kills = []
        self.stop = False

    def obs(self):
        if self.cl is None:
            import obsws_python as obs
            self.cl = obs.ReqClient(host=self.cfg["obs_host"], port=int(self.cfg["obs_port"]),
                                    password=self.cfg["obs_password"] or None, timeout=3)
            self.status.update(obs="connected", error="")
        return self.cl

    def sources(self):
        if self.replay:
            return ["(replay test frames)"]
        try:
            return [i["inputName"] for i in self.obs().get_input_list().inputs]
        except Exception as e:
            self.status.update(obs="error", error=friendly(e))
            self.cl = None
            return []

    def set_names(self, names):
        self.cfg["names"] = names
        self.reader.names = [norm(n) for n in names]

    def run(self):
        times = []
        while not self.stop:
            period = 1.0 / max(1, float(self.cfg["fps"]))
            t0 = time.time()
            try:
                if self.status["paused"] or not (self.cfg["source"] or self.replay):
                    if self.status["obs"] != "error":     # keep a connection error visible until it's fixed
                        self.status["error"] = "" if self.cfg["source"] else "Step 2: pick the source that shows WARDOGS and press Save."
                    time.sleep(0.5)
                    continue
                raw = self.grab()
                frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                full = f"{frame.shape[1]}x{frame.shape[0]}"
                big = frame                       # the reader gets this full size (it reads the small feed text from it)
                if frame.shape[0] > 720:
                    # shrink here with pixel averaging to 720 tall (the game sizes its HUD by height).
                    # OBS's own scaler drops pixels at 2:1 (1440p -> 720p) and chops up thin HUD text
                    frame = cv2.resize(frame, (round(frame.shape[1] * 720 / frame.shape[0]), 720),
                                       interpolation=cv2.INTER_AREA)
                    raw = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
                self.last_jpg = raw
                self.status["frame"] = f"{frame.shape[1]}x{frame.shape[0]}"
                self.status["native"] = full if not self.replay else self.status.get("native")
                for ev in self.reader.process(big, time.time()):
                    self.hub.send_threadsafe(ev)
                    self.clip_rule(ev)
                    self.log_event(ev)
                times = (times + [time.time() - t0])[-20:]
                self.status.update(obs="replay" if self.replay else "connected", ms=int(1000 * sum(times) / len(times)),
                                   fps=round(1 / max(period, sum(times) / len(times)), 1),
                                   frames=self.status["frames"] + 1, error="")
            except Exception as e:
                self.status.update(obs="error", error=friendly(e))
                self.cl = None
                traceback.print_exc()
                time.sleep(2)
                continue
            time.sleep(max(0.0, period - (time.time() - t0)))

    def grab(self):
        """One 1280x720 JPEG of the game: from OBS, or the next test frame in replay mode."""
        if self.replay:
            f = self.replay[self.replay_i % len(self.replay)]
            self.replay_i += 1
            with open(f, "rb") as fh:
                return fh.read()
        # full size (no width/height): any resolution works, we scale it down ourselves
        cl = self.obs()                            # connection problems surface here as before
        if not getattr(self, "small_only", False):
            try:
                r = cl.send("GetSourceScreenshot", {"sourceName": self.cfg["source"], "imageFormat": "jpg",
                                                    "imageCompressionQuality": 85}, raw=True)
                return base64.b64decode(r["imageData"].split(",", 1)[1])
            except Exception as e:
                if "OBSSDKRequestError" not in type(e).__name__:
                    raise
                self.small_only = True            # this OBS won't do full size: use its 1280x720 copy
        r = cl.get_source_screenshot(self.cfg["source"], "jpg", 1280, 720, 80)
        return base64.b64decode(r.image_data.split(",", 1)[1])

    def native(self):
        """Full-resolution PNG of the game source, exactly as OBS has it (no scaling)."""
        r = self.obs().send("GetSourceScreenshot", {"sourceName": self.cfg["source"], "imageFormat": "png"}, raw=True)
        return base64.b64decode(r["imageData"].split(",", 1)[1])

    def log_event(self, ev):
        """Every event of the session (minus the speed readouts) in logs/: lets us check a stream later."""
        if ev["type"] in ("pilot", "speed"):
            return
        try:
            if not getattr(self, "_log", None):
                os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
                old = sorted(glob.glob(os.path.join(HERE, "logs", "events-*.jsonl")))
                for f in old[:-19]:                # keep the last 20 sessions
                    os.remove(f)
                self._log = open(os.path.join(HERE, "logs", time.strftime("events-%Y%m%d-%H%M%S.jsonl")),
                                 "a", encoding="utf-8")
            self._log.write(json.dumps(dict(ev, clock=time.strftime("%H:%M:%S")), ensure_ascii=False) + "\n")
            self._log.flush()
        except Exception:
            pass

    def clip_rule(self, ev):
        """Save the OBS replay buffer ~8 s after a big moment (the reaction is the content)."""
        if not self.cfg.get("replay_buffer"):
            return
        now = time.time()
        big = False
        if ev["type"] == "selfkill":
            big = True
        elif ev["type"] == "kill":
            self.kills = [k for k in self.kills if now - k < 60] + [now]
            big = (ev.get("dist") or 0) >= 100 or len(self.kills) >= 3
        if big:
            threading.Timer(8.0, self._save_replay).start()

    def _save_replay(self):
        try:
            self.obs().save_replay_buffer()
            self.hub.send_threadsafe(dict(type="clip_saved", t=time.time()))
        except Exception as e:
            self.hub.send_threadsafe(dict(type="clip_failed", error=str(e), t=time.time()))


def make_app(cfg, replay=None):
    hub = Hub()
    cap = Capture(hub, cfg, replay=replay)
    app = web.Application()

    async def page(name):
        with open(os.path.join(HERE, "web", name), encoding="utf-8") as f:
            return web.Response(text=f.read(), content_type="text/html")

    async def index(_):
        return await page("control.html")

    async def overlay(_):
        return await page("overlay.html")

    async def ws_handler(request):
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        hub.clients.add(ws)
        await ws.send_str(json.dumps(dict(type="hello", session=hub.session)))
        # a page that (re)connects mid-flight needs to know what's going on right now
        await ws.send_str(json.dumps(dict(type="state", t=time.time(), **cap.reader.state)))
        async for msg in ws:
            if msg.type == WSMsgType.ERROR:
                break
            if msg.type == WSMsgType.TEXT and '"overlay"' in msg.data:
                hub.overlays.add(ws)
        hub.clients.discard(ws)
        hub.overlays.discard(ws)
        return ws

    async def status(_):
        return web.json_response(dict(cap.status, names=cfg["names"], fps_target=cfg["fps"],
                                      replay_buffer=cfg["replay_buffer"], clients=len(hub.clients),
                                      overlays=len(hub.overlays), pilot_check=cap.reader.tpl.last))

    async def sources(_):
        return web.json_response(await asyncio.to_thread(cap.sources))

    async def set_config(request):
        body = await request.json()
        for k in ("source", "fps", "replay_buffer", "obs_host", "obs_port", "obs_password"):
            if k in body:
                cfg[k] = body[k]
        if "names" in body:
            cap.set_names([n.strip() for n in body["names"] if n.strip()])
        if "paused" in body:
            cap.status["paused"] = bool(body["paused"])
        if any(k in body for k in ("obs_host", "obs_port", "obs_password")):
            cap.cl = None
        cap.status["source"] = cfg["source"]
        save_config(cfg)
        return web.json_response(dict(ok=True))

    async def test_event(request):
        ev = await request.json()
        ev.setdefault("t", time.time())
        ev["test"] = True
        await hub.send(ev)
        return web.json_response(dict(ok=True))

    def check_updates():
        """Ask GitHub for the newest version now and then (the control panel shows an Update button)."""
        while True:
            try:
                v, notes = updater.latest()
                upd.update(latest=v, notes=notes, available=updater.vt(v) > updater.vt(updater.current()))
            except Exception:
                pass
            time.sleep(20 * 60)

    upd = dict(current=updater.current(), latest=None, notes="", available=False)

    async def version(_):
        return web.json_response(upd)

    async def update(_):
        """Hand over to launcher.py: exit code 3 = install the update and start again."""
        if not upd["available"]:
            return web.json_response(dict(ok=False, error="Already up to date."))
        asyncio.get_running_loop().call_later(0.5, os._exit, 3)
        return web.json_response(dict(ok=True))

    async def reset(_):
        """Fresh start for a new stream: forget test clicks, kills, cash baseline, everything."""
        cap.reader = Reader(names=cfg["names"])
        hub.log = []
        await hub.send(dict(type="reset", t=time.time()))
        hub.log = []
        return web.json_response(dict(ok=True))

    async def open_logs(_):
        """Open the folder with this PC's session logs (one file per START.bat run) to send to Kevin."""
        folder = os.path.join(HERE, "logs")
        os.makedirs(folder, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(folder)
        return web.json_response(dict(ok=True, folder=folder))

    async def debug(_):
        """Save what the reader sees right now so benged can send it to Kevin, then open the folder."""
        folder = os.path.join(HERE, "debug", time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(folder, exist_ok=True)
        info = dict(status=cap.status, state=cap.reader.state, scores=cap.reader.tpl.last,
                    version=updater.current(), recent=hub.log[-80:])
        try:
            png = await asyncio.to_thread(cap.native)
            with open(os.path.join(folder, "game-full-size.png"), "wb") as f:
                f.write(png)
        except Exception as e:
            info["native_error"] = friendly(e)
        if cap.last_jpg:
            with open(os.path.join(folder, "reader-view.jpg"), "wb") as f:
                f.write(cap.last_jpg)
        with open(os.path.join(folder, "details.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=1, default=str)
        if sys.platform == "win32":
            os.startfile(folder)
        return web.json_response(dict(ok=True, folder=folder))

    async def snapshot(_):
        if not cap.last_jpg:
            return web.Response(status=404, text="no frame yet")
        img = cv2.imdecode(np.frombuffer(cap.last_jpg, np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        for name, (x0, y0, x1, y1) in REGIONS.items():
            cv2.rectangle(img, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), (0, 220, 255), 1)
            cv2.putText(img, name, (int(x0 * w) + 3, int(y0 * h) + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1)
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return web.Response(body=jpg.tobytes(), content_type="image/jpeg", headers={"Cache-Control": "no-store"})

    async def recent(_):
        return web.json_response(hub.log[-60:])

    async def on_start(_):
        hub.loop = asyncio.get_running_loop()
        cap.start()
        threading.Thread(target=check_updates, daemon=True).start()

    app.router.add_get("/", index)
    app.router.add_get("/overlay", overlay)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/api/status", status)
    app.router.add_get("/api/sources", sources)
    app.router.add_post("/api/config", set_config)
    app.router.add_post("/api/test", test_event)
    app.router.add_post("/api/reset", reset)
    app.router.add_post("/api/debug", debug)
    app.router.add_post("/api/logs", open_logs)
    app.router.add_get("/api/version", version)
    app.router.add_post("/api/update", update)
    app.router.add_get("/api/snapshot.jpg", snapshot)
    app.router.add_get("/api/recent", recent)
    app.on_startup.append(on_start)
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", help="folder of 1280x720 .jpg frames to play instead of OBS (testing)")
    args = ap.parse_args()
    cfg = load_config()
    if not os.path.exists(CONFIG_PATH):
        save_config(cfg)
    print(f"\n  Benged Network is running.\n  Control panel: http://localhost:{cfg['port']}/\n"
          f"  OBS browser source: http://localhost:{cfg['port']}/overlay  (1920 x 1080)\n")
    app = make_app(cfg, replay=args.replay)
    try:   # "localhost" can mean IPv4 or IPv6 depending on the program asking; answer on both
        web.run_app(app, host=["127.0.0.1", "::1"], port=int(cfg["port"]), print=None)
    except OSError:
        web.run_app(make_app(cfg, replay=args.replay), host="127.0.0.1", port=int(cfg["port"]), print=None)
