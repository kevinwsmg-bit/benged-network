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
from rl.stats import Translator                   # noqa: E402
from rl import setup as rl_setup                  # noqa: E402
import updater                                    # noqa: E402

CONFIG_PATH = os.path.join(HERE, "config.json")
DEFAULTS = {
    "obs_host": "localhost",
    "obs_port": 4455,
    "obs_password": "",
    "source": "",              # name of the WARDOGS game capture source in OBS
    "names": ["Benged"],       # in-game names to treat as him (add alt accounts here)
    "fps": 2,
    "port": 8765,
    "replay_buffer": False,    # save OBS replay buffer on big moments (crash, long shot, 3+ streak)
    "game": "wardogs",         # wardogs (reads the screen) | rocketleague (the game's own Stats API)
    "setup": "auto",           # where he plays WARDOGS: auto | home (1440x900 fullscreen) | buddy (1920x1080 windowed)
    "rl_port": 49124,          # Rocket League Stats API WebSocket port (WebPort in TAStatsAPI.ini)
}


def load_config():
    cfg = dict(DEFAULTS)
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg.update(json.load(f))
    # 2 checks a second is plenty (every event lasts several seconds) and halves the screenshots OBS
    # has to make; older installs saved 3
    cfg["fps"] = min(float(cfg.get("fps") or 2), 2)
    return cfg


def go_easy_on_the_game():
    """Run below normal priority so WARDOGS and OBS always get the CPU first; we only need leftovers."""
    if sys.platform == "win32":
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.GetCurrentProcess.restype = ctypes.c_void_p      # a 64-bit handle (plain int would mangle it)
            k.SetPriorityClass(ctypes.c_void_p(k.GetCurrentProcess()), 0x4000)   # BELOW_NORMAL
        except Exception:
            pass


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
        self.reader = Reader(names=cfg["names"], setup=cfg.get("setup", "auto"))
        self.status = dict(obs="connecting", source=cfg["source"], fps=0.0, ms=0, frames=0, error="", paused=False)
        self.last_frame = self.last_raw = None
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
                if self.cfg.get("game", "wardogs") != "wardogs" and not self.replay:
                    self.status["error"] = ""
                    time.sleep(0.5)            # Rocket League talks to us directly: no screenshots needed
                    continue
                if self.status["paused"] or not (self.cfg["source"] or self.replay):
                    if self.status["obs"] != "error":     # keep a connection error visible until it's fixed
                        self.status["error"] = "" if self.cfg["source"] else "Step 2: pick the source that shows WARDOGS and press Save."
                    time.sleep(0.5)
                    continue
                raw = self.grab()
                t_obs = time.time() - t0              # how long OBS took to hand over the picture
                frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                full = f"{frame.shape[1]}x{frame.shape[0]}"
                big = frame                       # the reader gets this full size (it reads the small feed text from it)
                if frame.shape[0] > 720:
                    # shrink here with pixel averaging to 720 tall (the game sizes its HUD by height).
                    # OBS's own scaler drops pixels at 2:1 (1440p -> 720p) and chops up thin HUD text
                    frame = cv2.resize(frame, (round(frame.shape[1] * 720 / frame.shape[0]), 720),
                                       interpolation=cv2.INTER_AREA)
                    raw = None                        # the control panel's preview is encoded only when asked for
                self.last_frame, self.last_raw = frame, raw
                self.status["frame"] = f"{frame.shape[1]}x{frame.shape[0]}"
                self.status["native"] = full if not self.replay else self.status.get("native")
                for ev in self.reader.process(big, time.time()):
                    self.hub.send_threadsafe(ev)
                    self.clip_rule(ev)
                    self.log_event(ev)
                times = (times + [time.time() - t0])[-20:]
                obs_t = (getattr(self, "_obs_t", []) + [t_obs])[-20:]
                self._obs_t = obs_t
                obs_ms = int(1000 * sum(obs_t) / len(obs_t))
                self.status.update(obs_ms=obs_ms, read_ms=max(0, int(1000 * sum(times) / len(times)) - obs_ms))
                if self.status["frames"] % 120 == 0:   # every minute or so: timing into the stream log
                    self.log_event(dict(type="perf", obs_ms=obs_ms, read_ms=self.status["read_ms"],
                                        picture=full, profile=self.reader.profile, t=time.time()))
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

    @property
    def last_jpg(self):
        """The latest picture as JPEG (control-panel preview / debug), encoded on demand."""
        if self.last_raw is not None:
            return self.last_raw
        if self.last_frame is None:
            return None
        return cv2.imencode(".jpg", self.last_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()

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
    rl = dict(tr=Translator(cfg["names"]), connected=False, last=None, error="", events=0)
    app = web.Application()

    async def page(name):
        with open(os.path.join(HERE, "web", name), encoding="utf-8") as f:
            return web.Response(text=f.read(), content_type="text/html")

    async def index(_):
        return await page("control.html")

    async def overlay(_):
        # one OBS URL for every game: the page matches the game picked in the control panel
        return await page("rl.html" if cfg.get("game") == "rocketleague" else "overlay.html")

    async def rl_send(evs):
        for e in evs:
            await hub.send(e)
            cap.log_event(e)
            rl["events"] += 1

    async def rl_client():
        """Rocket League's Stats API: connect while Rocket League is the picked game, retry quietly."""
        import aiohttp
        while True:
            if cfg.get("game") != "rocketleague":
                rl["connected"] = False
                await asyncio.sleep(1)
                continue
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.ws_connect(f"ws://127.0.0.1:{int(cfg.get('rl_port', 49124))}", heartbeat=None) as ws:
                        rl.update(connected=True, error="")
                        async for m in ws:
                            if m.type == WSMsgType.TEXT:
                                rl["last"] = time.time()
                                await rl_send(rl["tr"].feed(m.data))
                            elif m.type in (WSMsgType.CLOSED, WSMsgType.ERROR):
                                break
                            if cfg.get("game") != "rocketleague":
                                break
            except Exception:
                rl["error"] = "Waiting for Rocket League (start a match; stats must be switched on once)."
            rl["connected"] = False
            await asyncio.sleep(3)

    async def rl_setup_api(_):
        """Switch on Rocket League's Stats API in every install we can find."""
        installs = await asyncio.to_thread(rl_setup.find_installs)
        if not installs:
            return web.json_response(dict(ok=False, error="Couldn't find Rocket League on this PC. Tell Kevin where it's installed."))
        results = []
        for i in installs:
            try:
                st = await asyncio.to_thread(rl_setup.enable, i)
                results.append(dict(install=i, ok=True, **st))
            except PermissionError:
                results.append(dict(install=i, ok=False, path=rl_setup.ini_path(i),
                                    error="Windows blocked the change. Close START.bat, right-click it, Run as administrator, and press the button again."))
        return web.json_response(dict(ok=any(r["ok"] for r in results), results=results))

    async def rl_sim(request):
        """Fake Rocket League moments for testing the overlay without a match."""
        kind = (await request.json()).get("kind", "")
        me = (cfg["names"] or ["Benged"])[0]
        tr = rl["tr"]
        mine = 0

        def upd(blue, orange, clock=180, ot=False):
            return {"Event": "UpdateState", "Data": {"MatchGuid": "SIM", "Players": [
                {"Name": me, "TeamNum": mine, "Goals": 1, "Saves": 2, "Shots": 3, "Score": 300, "Demos": 1},
                {"Name": "TheMailMan", "TeamNum": mine}, {"Name": "Orange #1", "TeamNum": 1}, {"Name": "Orange #2", "TeamNum": 1}],
                "Game": {"Teams": [{"TeamNum": 0, "Score": blue}, {"TeamNum": 1, "Score": orange}], "TimeSeconds": clock, "bOvertime": ot}}}

        def P(n, t):
            return {"Name": n, "TeamNum": t}

        def goal(team, scorer, last, speed, assister=None):
            d = {"GoalSpeed": speed, "Scorer": P(scorer, team), "BallLastTouch": {"Player": P(last[0], last[1])}}
            if assister:
                d["Assister"] = P(assister, team)
            return [(0, {"Event": "GoalScored", "Data": d}), (1.2, {"Event": "GoalReplayStart", "Data": {}}),
                    (6.5, {"Event": "GoalReplayWillEnd", "Data": {}}), (1.5, {"Event": "GoalReplayEnd", "Data": {}}),
                    (0.5, {"Event": "CountdownBegin", "Data": {}}), (3, {"Event": "RoundStarted", "Data": {}})]

        known = tr.my_team is not None
        steps = [(0, upd(tr.score[0] if known else 0, tr.score[1] if known else 0, tr.clock if known else 180))]
        if kind == "goal_me":
            steps += goal(0, me, (me, 0), [38, 64, 97, 118, 142, 171][int(time.time()) % 6], "TheMailMan")
        elif kind == "goal_team":
            steps += goal(0, "TheMailMan", ("TheMailMan", 0), 91, me)
        elif kind == "goal_against":
            steps += goal(1, "Orange #1", ("Orange #1", 1), 104)
        elif kind == "own_goal":
            steps += goal(1, "Orange #2", (me, 0), 58)
        elif kind == "refund":
            steps += goal(0, me, (me, 0), 120)
            steps += [(1, upd(tr.score[0] + 1, tr.score[1], max(0, tr.clock - 9)))] + goal(1, "Orange #1", ("Orange #1", 1), 99)
        elif kind == "demoed":
            steps += [(0, {"Event": "StatfeedEvent", "Data": {"EventName": "Demolish", "Type": "Demolition",
                       "MainTarget": P("Orange #2", 1), "SecondaryTarget": P(me, 0)}})]
        elif kind == "demo_given":
            steps += [(0, {"Event": "StatfeedEvent", "Data": {"EventName": "Demolish", "Type": "Demolition",
                       "MainTarget": P(me, 0), "SecondaryTarget": P("Orange #1", 1)}})]
        elif kind in ("save", "epic_save"):
            steps += [(0, {"Event": "StatfeedEvent", "Data": {"EventName": "EpicSave" if kind == "epic_save" else "Save",
                       "Type": "Epic Save" if kind == "epic_save" else "Save", "MainTarget": P(me, 0)}})]
        elif kind in ("win", "loss"):
            steps += [(0, {"Event": "MatchEnded", "Data": {"WinnerTeamNum": 0 if kind == "win" else 1}}),
                      (2, {"Event": "PodiumStart", "Data": {}})]
        elif kind == "kickoff":
            steps += [(0, {"Event": "CountdownBegin", "Data": {}}), (3, {"Event": "RoundStarted", "Data": {}})]
        elif kind == "new_match":
            steps = [(0, {"Event": "MatchCreated", "Data": {"MatchGuid": "SIM"}}), (0, upd(0, 0, 300)),
                     (0, {"Event": "CountdownBegin", "Data": {}}), (3, {"Event": "RoundStarted", "Data": {}})]
        else:
            return web.json_response(dict(ok=False, error="unknown kind"))

        async def play():
            for delay, msg in steps:
                await asyncio.sleep(delay)
                await rl_send(tr.feed(msg))
        asyncio.get_running_loop().create_task(play())
        return web.json_response(dict(ok=True))

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
        return web.json_response(dict(cap.status, names=cfg["names"], fps_target=cfg["fps"], game=cfg.get("game", "wardogs"),
                                      setup=cfg.get("setup", "auto"), profile=cap.reader.profile, picture=cap.reader.picture,
                                      rl=dict(connected=rl["connected"], error=rl["error"], events=rl["events"],
                                              last=round(time.time() - rl["last"], 1) if rl["last"] else None,
                                              record=[rl["tr"].wins, rl["tr"].losses]),
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
        if body.get("setup") in ("auto", "home", "buddy"):
            cfg["setup"] = body["setup"]
            cap.reader.setup = body["setup"]          # takes effect on the next picture
        if body.get("game") in ("wardogs", "rocketleague") and body["game"] != cfg.get("game"):
            cfg["game"] = body["game"]
            rl["tr"] = Translator(cfg["names"])
            await hub.send(dict(type="reset", t=time.time()))     # overlays reload into the other game's look
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
        cap.reader = Reader(names=cfg["names"], setup=cfg.get("setup", "auto"))
        rl["tr"] = Translator(cfg["names"])
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
        asyncio.get_running_loop().create_task(rl_client())

    app.router.add_get("/", index)
    app.router.add_get("/overlay", overlay)
    os.makedirs(os.path.join(HERE, "web", "assets"), exist_ok=True)
    app.router.add_static("/assets", os.path.join(HERE, "web", "assets"))   # pictures + sounds the overlay uses
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
    app.router.add_post("/api/rl/setup", rl_setup_api)
    app.router.add_post("/api/rl/sim", rl_sim)
    app.on_startup.append(on_start)
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", help="folder of 1280x720 .jpg frames to play instead of OBS (testing)")
    args = ap.parse_args()
    go_easy_on_the_game()
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
