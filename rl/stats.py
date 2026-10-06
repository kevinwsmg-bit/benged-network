"""Rocket League: the game's own Stats API -> overlay events (BNSN, Benged Network Sports).

Rocket League sends JSON over a local WebSocket while a match runs (official feature, switched on
in TAGame\\Config\\TAStatsAPI.ini, see setup.py). Every message is {"Event": name, "Data": {...}}.
Nothing reads the screen or the game's memory: this costs almost no CPU.

What we get while PLAYING (not spectating): every player's name, team, goals, shots, assists, saves,
demos and score; the clock and team scores; the ball's speed and which team touched it last; and
events (GoalScored with speed, scorer, assister and last touch; StatfeedEvent: saves, demolitions,
shots...; GoalReplayStart/WillEnd/End; CountdownBegin; RoundStarted; MatchEnded; PodiumStart).
Boost amounts and car speeds are spectator-only, so nothing here depends on them.

The Translator is pure (no sockets): feed it messages, it returns the events the overlay needs.
"""
import json
import time

# goal speed: the API doc says Unreal units/s, its example (87.3) looks like km/h; the game's own goal
# banner shows km/h. Anything under 300 is treated as km/h, bigger numbers are converted.
def kmh(v):
    v = float(v or 0)
    return round(v if v < 300 else v * 0.036)


class Translator:
    def __init__(self, names=("Benged",)):
        self.names = [n.lower() for n in names if n]
        self.reset_session()
        self.reset_match()

    # ------------------------------------------------------------ state
    def reset_session(self):
        """New stream (START.bat or the Reset button): the night's records start over."""
        self.wins = self.losses = 0
        self.streak = 0                  # +n = n wins in a row, -n = n losses in a row
        self.night = dict(goals=0, own_goals=0, demos_given=0, demoed=0, saves=0, refunds=0)

    def reset_match(self):
        self.guid = None
        self.my_team = None
        self.me = None                   # his name exactly as the game shows it
        self.score = [0, 0]
        self.clock, self.ot = 300, False
        self.round_live = False
        self.last_my_goal = None          # (match clock, game time) of his team's last goal: refund check
        self.match_stats = {}

    def is_me(self, name):
        n = (name or "").lower()
        return any(x in n for x in self.names)

    def _who(self, p):
        p = p or {}
        name = p.get("Name") or p.get("PlayerName") or ""
        return dict(name=name, team=p.get("TeamNum"), me=self.is_me(name))

    # ------------------------------------------------------------ feed
    def feed(self, msg):
        if isinstance(msg, (bytes, str)):
            try:
                msg = json.loads(msg)
            except ValueError:
                return []
        ev, d = msg.get("Event"), msg.get("Data") or {}
        if isinstance(d, str):                  # some builds double-encode Data
            try:
                d = json.loads(d)
            except ValueError:
                d = {}
        h = getattr(self, "_on_" + (ev or ""), None)
        out = h(d) if h else []
        for e in out:
            e.setdefault("t", time.time())
        return out

    def _on_UpdateState(self, d):
        out = []
        g = d.get("Game") or {}
        guid = d.get("MatchGuid")
        if guid and guid != self.guid:
            self.guid = guid
        for p in d.get("Players") or []:
            if self.is_me(p.get("Name")):
                self.me, team = p.get("Name"), p.get("TeamNum")
                if team != self.my_team:
                    self.my_team = team
                    out.append(dict(type="rl_team", team=team))
                self.match_stats = {k: p.get(k, 0) for k in ("Score", "Goals", "Shots", "Assists", "Saves", "Touches", "Demos")}
        teams = g.get("Teams") or []
        score = [0, 0]
        for t in teams:
            if t.get("TeamNum") in (0, 1):
                score[t["TeamNum"]] = t.get("Score", 0)
        clock, ot = g.get("TimeSeconds", self.clock), bool(g.get("bOvertime"))
        if score != self.score or clock != self.clock or ot != self.ot:
            self.score, self.clock, self.ot = score, clock, ot
            out.append(dict(type="rl_score", blue=score[0], orange=score[1], clock=clock, ot=ot,
                            my_team=self.my_team, players=len(d.get("Players") or [])))
        return out

    def _on_MatchCreated(self, d):
        self.reset_match()
        self.guid = d.get("MatchGuid")
        return [dict(type="rl_match", phase="created")]

    def _on_MatchInitialized(self, d):
        return [dict(type="rl_match", phase="initialized")]

    def _on_CountdownBegin(self, d):
        self.round_live = False
        return [dict(type="rl_dead", why="countdown")]

    def _on_RoundStarted(self, d):
        self.round_live = True
        return [dict(type="rl_live")]          # GO: every card must be gone now

    def _on_GoalReplayStart(self, d):
        return [dict(type="rl_dead", why="replay")]

    def _on_GoalReplayWillEnd(self, d):
        return [dict(type="rl_replay_ending")]

    def _on_GoalReplayEnd(self, d):
        return [dict(type="rl_replay_end")]

    def _on_GoalScored(self, d):
        self.round_live = False
        scorer = self._who(d.get("Scorer"))
        assister = self._who(d.get("Assister")) if d.get("Assister") else None
        last = self._who((d.get("BallLastTouch") or {}).get("Player"))
        team = scorer["team"]
        if team in (0, 1):
            self.score[team] += 1
        mine = self.my_team is not None and team == self.my_team
        # own goal: he touched it last and it went in for the other team
        own_goal = last["me"] and self.my_team is not None and team is not None and team != self.my_team
        speed = kmh(d.get("GoalSpeed"))
        e = dict(type="rl_goal", speed=speed, scorer=scorer["name"], scorer_me=scorer["me"],
                 assister=assister["name"] if assister else "", assister_me=bool(assister and assister["me"]),
                 team=team, mine=mine, against=(self.my_team is not None and not mine), own_goal=own_goal,
                 blue=self.score[0], orange=self.score[1], clock=self.clock, ot=self.ot, my_team=self.my_team,
                 round_secs=round(float(d.get("GoalTime") or 0), 1))
        if scorer["me"]:
            self.night["goals"] += 1
        if own_goal:
            self.night["own_goals"] += 1
        # celebration refund: his team scored, the other team answered within 20 s of play
        now_clock = self.clock if not self.ot else -1
        if mine:
            self.last_my_goal = dict(clock=self.clock, ot=self.ot, me=scorer["me"])
        elif self.last_my_goal and self.last_my_goal["me"] and not self.ot and not self.last_my_goal["ot"] \
                and 0 <= self.last_my_goal["clock"] - now_clock <= 20:
            e["refund"] = True
            self.night["refunds"] += 1
            self.last_my_goal = None
        e["night"] = dict(self.night)
        return [e]

    def _on_StatfeedEvent(self, d):
        main = self._who(d.get("MainTarget"))
        sec = self._who(d.get("SecondaryTarget")) if d.get("SecondaryTarget") else None
        name = d.get("EventName") or ""
        label = d.get("Type") or name
        out = []
        if name == "Demolish" or "Demol" in label:
            if sec and sec["me"]:
                self.night["demoed"] += 1
                out.append(dict(type="rl_demo", victim_me=True, attacker=main["name"],
                                attacker_team=main["team"], count=self.night["demoed"]))
            elif main["me"]:
                self.night["demos_given"] += 1
                out.append(dict(type="rl_demo", attacker_me=True, victim=sec["name"] if sec else "",
                                count=self.night["demos_given"]))
            return out
        if main["me"] and ("Save" in name or "Save" in label):
            self.night["saves"] += 1
            out.append(dict(type="rl_save", epic="Epic" in name or "Epic" in label, count=self.night["saves"]))
        out.append(dict(type="rl_stat", name=name, label=label, who=main["name"], me=main["me"], team=main["team"]))
        return out

    def _on_CrossbarHit(self, d):
        last = self._who((d.get("BallLastTouch") or {}).get("Player"))
        return [dict(type="rl_crossbar", me=last["me"], speed=kmh(d.get("BallSpeed")))]

    def _on_MatchEnded(self, d):
        self.round_live = False
        won = self.my_team is not None and d.get("WinnerTeamNum") == self.my_team
        if self.my_team is None:
            return [dict(type="rl_match_end", won=None)]
        if won:
            self.wins += 1
            self.streak = self.streak + 1 if self.streak > 0 else 1
        else:
            self.losses += 1
            self.streak = self.streak - 1 if self.streak < 0 else -1
        mine, theirs = self.score[self.my_team], self.score[1 - self.my_team]
        return [dict(type="rl_match_end", won=won, streak=self.streak, wins=self.wins, losses=self.losses,
                     my_score=mine, their_score=theirs, ot=self.ot, stats=dict(self.match_stats),
                     night=dict(self.night))]

    def _on_PodiumStart(self, d):
        return [dict(type="rl_dead", why="podium")]

    def _on_MatchDestroyed(self, d):
        self.reset_match()
        return [dict(type="rl_match", phase="left")]
