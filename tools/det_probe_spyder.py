#!/usr/bin/env python3
"""
Determinism probe for Tuxemon -- SPYDER scenario.

Unlike ``tools/det_probe.py`` (which teleports the player onto
``cotton_town.tmx`` and force-sets ``badge``), this probe starts the game the
way a player does:

    start_tuxemon.tmx  ->  scenario menu -> "TUXEMON: SPYDER AND THE CATHEDRAL"
                       ->  appearance menu -> pronoun menu
                       ->  transition_teleport to spyder_bedroom.tmx (4, 4)

No ``teleport`` action is executed by the harness for the ``intro``,
``journey`` and ``gymwalk`` scenarios; every map change is produced by the
game's own events.  The ``gym`` scenario *does* teleport (see SCENARIOS below);
it exists only to exercise the gym interior and its numbers are NOT a baseline.

``gymwalk`` is the full route, boot -> spyder_leather_gym.tmx, seven map
transitions, zero teleports::

    start_tuxemon -> (scenario/appearance/pronoun menus)
                  -> spyder_bedroom -> spyder_downstairs -> spyder_paper_town
                  -> (starter monster + mandatory rival battle vs spyder_billie)
                  -> spyder_route1     (up)
                  -> spyder_cotton_town(up)
                  -> spyder_route2     (right)
                  -> spyder_citypark   (up)
                  -> spyder_leather_town (left)
                  -> spyder_leather_gym  (up)

The harness decides the step count, so a digest mismatch is engine divergence
and not observer jitter.

Usage:
    TUXEMON_DETERMINISTIC=1 python tools/det_probe_spyder.py \
        --scenario intro --out run1.json
    TUXEMON_DETERMINISTIC=1 python tools/det_probe_spyder.py \
        --scenario gymwalk --out gymwalk1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# The determinism shim must be installed before anything else grabs a
# reference to time.time / time.perf_counter.
from tuxemon import determinism  # noqa: E402

determinism.install()

import pygame  # noqa: E402

from tuxemon.client import LocalPygameClient  # noqa: E402
from tuxemon.db import Direction  # noqa: E402
from tuxemon.prepare import pygame_init  # noqa: E402
from tuxemon.session import local_session  # noqa: E402
from tuxemon.startup_state_machine import StartupStateMachine  # noqa: E402
from tuxemon.user_config import CONFIG  # noqa: E402

FIXED_DT = 1.0 / 60.0
PLAN_FACING = Direction.DOWN

# States the driver must simply wait out (cutscene / transition / combat
# animation).  Pressing keys during them does nothing useful.
WAIT_STATES = ("SinkState", "CombatState", "TeleporterState")

# States the driver drives with CONFIRM.  Anything whose class name ends in
# "MenuState" is treated the same way.
MENU_STATES = (
    "ChoiceState",
    "ChoiceMonster",
    "ChoiceNpc",
    "InputMenu",
    "DialogState",
    "JournalInfoState",
    "MonsterInfoState",
    "Menu",
    # Post-XP states.  These only appear once the party actually *wins* a
    # fight, so the `intro`/`journey` scenarios never reached them.  None of
    # their class names ends in "MenuState", so before they were listed here
    # top_menu() skipped straight past them to the CombatState underneath,
    # classified the stack as "__wait__" and the driver sat on its hands
    # forever -- a hard deadlock in the middle of combat.
    "LevelUpSummaryState",   # tuxemon/states/level_up.py:21
    "MonsterMovesState",     # tuxemon/states/monster_moves.py:34 (forget a move)
    "EvolutionState",        # tuxemon/states/evolution.py:21
    "EvolutionTransition",   # tuxemon/states/transition_evolution.py:36
)

# --------------------------------------------------------------------------
# synthetic keyboard shim
#
# pygame_menu gates *menu navigation* (arrow keys) behind
# pygame_menu.utils.check_key_pressed_valid(), which asks SDL whether any
# physical key is down (``True in pygame.key.get_pressed()``).  Events created
# with pygame.event.post() never satisfy that, so arrow keys are silently
# dropped in every pygame_menu-backed state (ChoiceMonster, ChoiceState, ...).
# Tuxemon itself never calls get_pressed(), so replacing it is safe and only
# affects the probe process.
# --------------------------------------------------------------------------
HELD: set[int] = set()


class _PressedView:
    def __contains__(self, value: object) -> bool:
        return bool(HELD) if value else True

    def __getitem__(self, key: int) -> bool:
        return key in HELD

    def __iter__(self):
        return iter([])


_PRESSED = _PressedView()
pygame.key.get_pressed = lambda: _PRESSED  # type: ignore[assignment]


def post_key(ev_type: int, key: int) -> None:
    if ev_type == pygame.KEYDOWN:
        HELD.add(key)
    else:
        HELD.discard(key)
    pygame.event.post(
        pygame.event.Event(ev_type, key=key, mod=0, unicode="", scancode=0)
    )


KEY = {
    "UP": pygame.K_UP,
    "DOWN": pygame.K_DOWN,
    "LEFT": pygame.K_LEFT,
    "RIGHT": pygame.K_RIGHT,
    "RET": pygame.K_RETURN,
    "ESC": pygame.K_ESCAPE,
}

# --------------------------------------------------------------------------
# scenarios
#
# A route entry is one of
#   ("<map>.tmx", (x, y))  -- walk there (BFS over the engine's own
#                             traversability rules, driven by arrow keys)
#   ("KEY", keycode, hold) -- tap a key for `hold` steps (used to set facing
#                             for `is char_facing player,<dir>` teleports and
#                             to press INTERACT)
#   ("WAIT", n)            -- hand the next n steps to the game
# --------------------------------------------------------------------------
INTRO_ROUTE: list = []

JOURNEY_ROUTE: list = [
    ("spyder_bedroom.tmx", (7, 2)),          # "Go Downstairs"
    ("spyder_downstairs.tmx", (4, 6)),       # "Go Outside" ...
    ("KEY", pygame.K_DOWN, 12),              # ... needs facing == down
    ("WAIT", 60),
    ("KEY", pygame.K_DOWN, 20),
    ("spyder_paper_town.tmx", (21, 9)),      # next to the Rockitten pen
    ("KEY", pygame.K_RIGHT, 10),             # face the pen
    ("WAIT", 20),
    ("KEY", pygame.K_RETURN, 6),             # INTERACT -> starter dialogue
    ("WAIT", 120),
    ("spyder_paper_town.tmx", (25, 8)),      # "First Fight" trigger (rival)
    ("WAIT", 900),
    ("spyder_paper_town.tmx", (13, 1)),      # north gate, opens after the fight
    ("KEY", pygame.K_UP, 20),
    ("WAIT", 40),
    ("spyder_route1.tmx", (21, 1)),          # tall grass -> wild encounters
    ("KEY", pygame.K_UP, 20),
    ("WAIT", 40),
    ("spyder_cotton_town.tmx", (22, 35)),
]

# The four hops past spyder_cotton_town.  The transition_teleport events
# themselves are ungated -- their only conditions are `is char_at player` and
# `is char_facing player,<dir>`:
#
#   spyder_cotton_town  (39, 29) facing right -> spyder_route2       (0, 9)
#   spyder_route2       (10, 0)  facing up    -> spyder_citypark     (10, 39)
#   spyder_citypark     (0, 12)  facing left  -> spyder_leather_town (39, 32)
#   spyder_leather_town (24, 20) facing up    -> spyder_leather_gym  (1, 10)
#
# ...but the *approach* to the first one is not.  The two tiles in front of
# cotton town's east gate, (38, 28) and (38, 29), carry the "Stop! Cotton"
# event (mods/tuxemon/maps/spyder_cotton_town.tmx, object "Stop! Cotton"):
#
#   cond1 = is char_at player
#   cond2 = not variable_set visitedcottoncafe:yes
#   act10 = char_stop player ; act11 = lock_controls
#   act50 = pathfind player,36,28 ...
#
# and (39, 27) / (39, 30) are solid collision, so (38, 28) / (38, 29) are the
# *only* way in.  The event never sets visitedcottoncafe itself, so it re-fires
# on every attempt and walks the player back west -- a genuine soft gate.
# `visitedcottoncafe:yes` is written by act52 of "First Visit to Cotton Cafe"
# in spyder_cotton_cafe.tmx, which is a `talk spyder_cottontown_hacker`
# behaviour.  So the route detours through the cafe and talks to the hacker.
# That is still pure walking + INTERACT; no teleport and no variable is forced.
GYMWALK_TAIL: list = [
    # -- cafe detour: unlock the east gate ------------------------------
    ("spyder_cotton_town.tmx", (31, 16)),    # cafe door
    ("KEY", pygame.K_UP, 20),
    ("WAIT", 90),
    ("spyder_cotton_cafe.tmx", (10, 5)),     # under the hacker at (10, 4)
    ("KEY", pygame.K_UP, 12),
    ("WAIT", 20),
    ("KEY", pygame.K_RETURN, 6),             # talk -> Tuxepedia cutscene
    ("WAIT", 1600),                          # ...which sets visitedcottoncafe
    ("spyder_cotton_cafe.tmx", (8, 11)),     # cafe exit
    ("KEY", pygame.K_DOWN, 20),
    ("WAIT", 90),
    # -- hop 4: cotton town -> route 2 ----------------------------------
    ("spyder_cotton_town.tmx", (39, 29)),    # east gate
    ("KEY", pygame.K_RIGHT, 20),
    ("WAIT", 90),
    # -- hop 5: route 2 -> city park ------------------------------------
    # entering at (0, 9) forces the "Billie encounter" trainer battle on
    # (1, 8) / (1, 9); it is unavoidable and one-shot (act90 sets
    # route2billie:yes).  The rest of route 2 is encounter grass.
    ("spyder_route2.tmx", (10, 0)),          # north gate
    ("KEY", pygame.K_UP, 20),
    ("WAIT", 90),
    # -- hop 6: city park -> leather town -------------------------------
    ("spyder_citypark.tmx", (0, 12)),        # west gate
    ("KEY", pygame.K_LEFT, 20),
    ("WAIT", 90),
    # -- hop 7: leather town -> gym -------------------------------------
    ("spyder_leather_town.tmx", (24, 20)),   # gym door
    ("KEY", pygame.K_UP, 20),
    ("WAIT", 150),
    ("spyder_leather_gym.tmx", (1, 8)),      # a step inside, proves arrival
]

GYMWALK_ROUTE: list = JOURNEY_ROUTE + GYMWALK_TAIL

GYM_ROUTE: list = [
    ("KEY", pygame.K_DOWN, 10),              # face Chad, who stands at (9, 9)
    ("WAIT", 20),
    ("KEY", pygame.K_RETURN, 6),             # talk -> start_battle chad/brad
    ("WAIT", 1500),
    ("spyder_leather_gym.tmx", (12, 8)),     # above Brad (12, 9)
    ("KEY", pygame.K_DOWN, 10),
    ("WAIT", 20),
    ("KEY", pygame.K_RETURN, 6),
    ("WAIT", 1500),
    ("spyder_leather_gym.tmx", (6, 3)),      # the scoreboard at (6, 2)
    ("KEY", pygame.K_UP, 10),
    ("WAIT", 20),
    ("KEY", pygame.K_RETURN, 6),
    ("WAIT", 200),
]

SCENARIOS = {
    # name: (route, default steps, teleport-to-map-or-None, wild policy)
    "intro": (INTRO_ROUTE, 2600, None, "run"),
    # 15000: the route reaches spyder_cotton_town around step 13.4k.
    "journey": (JOURNEY_ROUTE, 15000, None, "run"),
    # boot -> spyder_leather_gym.tmx, 7 map transitions, no teleports.
    #
    # The wild policy is "fight", not "run", and that is not a preference: the
    # "Billie encounter" event on spyder_route2 (1, 8)/(1, 9) is a *mandatory*
    # trainer battle with no Run entry, and losing it warps the player to the
    # teleport_faint point (spyder_bedroom 3,4, set by spyder_paper_town.tmx
    # act11).  A party that flees every wild encounter never levels and can
    # never clear it.  Fighting the route 1 grass on the way is what makes the
    # gate passable.
    "gymwalk": (GYMWALK_ROUTE, 120000, None, "fight"),
    # The gym is 7 map transitions from the bedroom; this scenario teleports
    # after the (real) intro and exists only to exercise the gym interior.
    # Declared in the output as "teleported": true.  NOT a baseline.
    "gym": (GYM_ROUTE, 16000, ("spyder_leather_gym.tmx", 9, 8), "run"),
}

# Game variables the report calls out explicitly (task-specified + the ones
# the spyder branch actually writes).
WATCHED_VARS = (
    "spyder_intro",
    "question_intro",
    "scenario_choice",
    "race_choice",
    "gender_choice",
    "pronoun_choice",
    "choice_phase",
    "myintrochoice",
    "areyousure",
    "intro_scoop",
    "dantefirst",
    "dantebin",
    "mymonchoice",
    "firstfightdue",
    "firstfightend",
    "battle_last_result",
    "battle_last_winner",
    "brad_points",
    "chad_points",
    "chadvsbrad",
    "run_attempts",
)

# Label of the "Run" entry of MainCombatMenuState.  Only wild encounters get
# one (MenuProfiles.default_monster_battle vs default_trainer_battle in
# tuxemon/combat/menu_visibility.py), so its presence *is* the wild/trainer
# discriminator and the probe does not have to reach into CombatSession.
def run_label() -> str:
    from tuxemon.locale.locale import T

    return T.translate("menu_run").upper()


def make_client() -> LocalPygameClient:
    context = pygame_init()
    config = CONFIG.copy()
    # straight-to-mod boot: no title screen, no splash, no CLI thread.  The
    # mod's own startup rules still run, so the scenario/appearance/pronoun
    # menus appear exactly as they do for a player.
    config.config_model.game.skip_titlescreen = True
    config.config_model.display.splash = False
    config.config_model.game.cli_enabled = False
    config.mods = ["tuxemon"]
    client = LocalPygameClient.create(config, context)
    local_session.set_client(client)
    StartupStateMachine(client, config).run()
    return client


class Driver:
    """
    Reactive input driver.

    Its decisions are a pure function of the game state, so in deterministic
    mode it emits the same key at the same step every run -- which the probe
    checks separately via ``tape_digest``.
    """

    DIRKEY = {
        (0, -1): pygame.K_UP,
        (0, 1): pygame.K_DOWN,
        (-1, 0): pygame.K_LEFT,
        (1, 0): pygame.K_RIGHT,
    }

    def __init__(
        self,
        client: LocalPygameClient,
        route: list,
        wild: str = "run",
    ) -> None:
        self.client = client
        # "run"   -- flee every wild encounter (fastest traversal)
        # "fight" -- fight them (levels the party up, much slower)
        self.wild = wild
        self.tape: list[tuple[int, str, int]] = []
        self.pending: dict[int, list[tuple[int, int]]] = {}
        self.cooldown = 0
        self.route_plan = list(route)
        self.goal_i = 0
        self.held: int | None = None
        self.stuck = 0
        self.route: list[tuple[int, int]] = []
        self.blocked: set[tuple[int, int]] = set()
        self.last_tile: tuple[int, int] | None = None
        self.stall = 0
        self.menu_name: str | None = None
        self.menu_age = 0
        # combat: how many cursor moves we have spent trying to reach "Run"
        self.combat_nav = 0
        self.run_label = run_label()
        # wrong-map recovery (a lost battle teleports the player to the last
        # set_teleport_faint destination, which is usually behind us)
        self.offmap = 0
        self.events: list[dict] = []

    # -- input plumbing ----------------------------------------------------
    def emit(self, step: int, key: int, hold: int = 4) -> None:
        self.tape.append((step, "down", key))
        self.tape.append((step + hold, "up", key))
        self.pending.setdefault(step, []).append((pygame.KEYDOWN, key))
        self.pending.setdefault(step + hold, []).append((pygame.KEYUP, key))

    def press(self, step: int, key: int | None) -> None:
        """Hold `key`, releasing whatever else is held (None releases all)."""
        if self.held == key:
            return
        if self.held is not None:
            self.tape.append((step, "up", self.held))
            post_key(pygame.KEYUP, self.held)
        self.held = key
        if key is not None:
            self.tape.append((step, "down", key))
            post_key(pygame.KEYDOWN, key)

    def step(self, step: int) -> None:
        self.decide(step)
        for ev_type, key in self.pending.pop(step, []):
            post_key(ev_type, key)

    # -- policy ------------------------------------------------------------
    def top_menu(self):
        for st in self.client.state_manager.active_states:
            nm = type(st).__name__
            if nm in WAIT_STATES:
                return "__wait__", st
            if nm in MENU_STATES or nm.endswith("MenuState"):
                return nm, st
            if nm == "WorldState":
                return None, None
        return None, None

    def decide(self, step: int) -> None:
        if self.cooldown > 0:
            self.cooldown -= 1
            return
        nm, st = self.top_menu()

        # anti-deadlock: a menu that refuses to close gets an ESCAPE
        if nm == self.menu_name:
            self.menu_age += 1
        else:
            self.menu_name = nm
            self.menu_age = 0
        if nm is not None and nm != "__wait__" and self.menu_age > 40:
            self.press(step, None)
            self.emit(step, pygame.K_ESCAPE)
            self.cooldown = 12
            self.menu_age = 0
            return

        if nm == "__wait__":
            self.press(step, None)
            return
        if nm == "MainCombatMenuState":
            self.press(step, None)
            self.emit(step, self.combat_key(st))
            self.cooldown = 10
            return
        if nm == "Menu" and "CombatState" in self.client.active_state_names:
            key = self.technique_key(st)
            if key is not None:
                self.press(step, None)
                self.emit(step, key)
                self.cooldown = 10
                return
        if nm == "ChoiceMonster":
            # 5 columns x 4 rows; per monster: image, name-button (opens the
            # journal), pick-button, vertical fill.  DOWN moves name -> pick.
            try:
                title = st.menu.get_selected_widget().get_title()
            except Exception:
                title = ""
            self.press(step, None)
            self.emit(
                step,
                pygame.K_RETURN if title == "Pick" else pygame.K_DOWN,
            )
            self.cooldown = 10
            return
        if nm == "InputMenu":
            # 68 items in 13 columns: 65 = backspace, 66 = END, 67 = RANDOM
            idx = st.selected_index
            self.press(step, None)
            if idx == 66:
                self.emit(step, pygame.K_RETURN)
                self.cooldown = 14
            elif idx == 65:
                self.emit(step, pygame.K_RIGHT)
                self.cooldown = 8
            else:
                self.emit(step, pygame.K_DOWN)
                self.cooldown = 8
            return
        if nm is not None:
            self.press(step, None)
            self.emit(step, pygame.K_RETURN)
            self.cooldown = 10
            return
        self.walk(step)

    # -- combat ------------------------------------------------------------
    @staticmethod
    def nav_key(cur: int, target: int, cols: int) -> int:
        """Exact grid step from `cur` towards `target` in a `cols`-wide menu."""
        if cur % cols < target % cols:
            return pygame.K_RIGHT
        if cur % cols > target % cols:
            return pygame.K_LEFT
        if cur // cols < target // cols:
            return pygame.K_DOWN
        return pygame.K_UP

    def technique_key(self, st) -> int | None:
        """
        The technique list pushed by MainCombatMenuState.open_technique_menu
        (tuxemon/states/combat_menus.py) is a plain ``Menu`` whose items carry
        the Technique as their game_object.  Confirming blindly always picks
        the *first* move, which for a level 5 rockitten is `ram`
        (power 1.5, accuracy 1.0) rather than `mudslide` (2.25 x 0.8 = 1.8
        expected).  Pick the best expected-damage move instead; returns None
        when the menu is not a technique list, so the caller falls back to the
        generic CONFIRM.
        """
        try:
            items = list(st.menu_items)
            cols = max(1, int(getattr(st, "columns", 1)))
            cur = int(st.selected_index)
        except Exception:
            return None

        best, best_score = None, None
        for i, it in enumerate(items):
            tech = it.game_object
            power = getattr(tech, "power", None)
            if power is None:
                return None  # not a technique menu
            if not it.enabled:
                continue
            score = float(power) * float(getattr(tech, "accuracy", 1.0) or 0.0)
            if best_score is None or score > best_score:
                best, best_score = i, score
        if best is None:
            return pygame.K_RETURN
        if best == cur or self.combat_nav > 8:
            self.combat_nav = 0
            return pygame.K_RETURN
        self.combat_nav += 1
        return self.nav_key(cur, best, cols)

    def combat_key(self, st) -> int:
        """
        Wild encounter  -> walk the cursor onto "Run" and confirm.
        Trainer battle  -> confirm whatever is selected ("Fight"), which is the
                           only way those battles end.

        Purely reactive: it reads ``selected_index`` back out of the live menu
        every time, so it converges no matter how the grid is laid out and it
        emits the same key at the same step on every deterministic run.
        """
        try:
            items = list(st.menu_items)
            cols = max(1, int(getattr(st, "columns", 1)))
            cur = int(st.selected_index)
        except Exception:
            return pygame.K_RETURN

        target = None
        if self.wild == "run":
            for i, it in enumerate(items):
                if (it.label or "").upper() == self.run_label and it.enabled:
                    target = i
                    break

        # trainer battle (no Run entry), already on Run, or the cursor refuses
        # to move: confirm.  The nav budget stops a mis-modelled grid from
        # deadlocking the probe, since MainCombatMenuState.escape_key_exits is
        # False and the generic ESCAPE watchdog cannot rescue it.
        if target is None or target == cur or self.combat_nav > 8:
            self.combat_nav = 0
            return pygame.K_RETURN

        self.combat_nav += 1
        return self.nav_key(cur, target, cols)

    # -- walking -----------------------------------------------------------
    def bfs(self, start, goal, facing):
        """Breadth-first route over the engine's own traversability rules."""
        cm = self.client.collision_manager.get_collision_map()
        prev = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur == goal:
                path = []
                while cur != start:
                    path.append(cur)
                    cur = prev[cur]
                path.reverse()
                return path
            for nxt in self.client.pathfinder.get_exits(cur, facing, cm, set()):
                if nxt in self.blocked or nxt in prev:
                    continue
                prev[nxt] = cur
                q.append(nxt)
        return []

    def _next_goal(self, step: int) -> None:
        self._set_goal(step, self.goal_i + 1)

    def _set_goal(self, step: int, index: int) -> None:
        self.goal_i = index
        self.press(step, None)
        self.stuck = 0
        self.route = []
        self.blocked = set()
        self.stall = 0
        self.offmap = 0

    def walk(self, step: int) -> None:
        if self.goal_i >= len(self.route_plan):
            self.press(step, None)
            return
        goal = self.route_plan[self.goal_i]
        if goal[0] == "KEY":
            self.press(step, None)
            self.emit(step, goal[1], hold=goal[2])
            self.cooldown = goal[2] + 3
            self.goal_i += 1
            return
        if goal[0] == "WAIT":
            self.press(step, None)
            self.cooldown = goal[1]
            self.goal_i += 1
            return

        goal_map, goal_tile = goal
        cur_map = self.client.get_map_name()
        p = local_session.player
        tile = tuple(p.tile_pos)

        if cur_map != goal_map:
            # The goal tile of a hop *is* the transition_teleport trigger, and
            # walking onto it from the required direction fires the teleport in
            # the same frame -- the player never stands on it, so `tile ==
            # goal_tile` never becomes true and the following ("KEY", dir) /
            # ("WAIT") entries are moot.  Skip ahead to the first remaining
            # waypoint that names the map we are actually on.
            fwd = [
                j
                for j in range(self.goal_i + 1, len(self.route_plan))
                if self.route_plan[j][0] == cur_map
            ]
            if fwd:
                self._set_goal(step, fwd[0])
                return
            # Recovery: a lost battle warps the player to the last
            # set_teleport_faint destination, which is a map we already left.
            # Rewind to that map's waypoint and walk the hop again rather than
            # standing still forever.
            self.offmap += 1
            if self.offmap > 180:
                back = [
                    j
                    for j in range(self.goal_i)
                    if self.route_plan[j][0] == cur_map
                ]
                if back:
                    self.events.append(
                        {"step": step, "kind": "rewind", "value": cur_map}
                    )
                    self._set_goal(step, back[-1])
                    return
                self.offmap = 0
            self.press(step, None)
            return
        if tile == goal_tile:
            self._next_goal(step)
            return

        if tile == self.last_tile:
            self.stall += 1
        else:
            self.last_tile = tile
            self.stall = 0
        if self.stall > 90:
            # the engine's traversability check allowed a tile the movement
            # code then refuses; blacklist it and replan
            if self.route:
                self.blocked.add(self.route[0])
            self.route = []
            self.stall = 0

        # Do NOT gate on p.moving: the direction key is held continuously, so
        # the player is "moving" nearly always and gating on it would freeze
        # the planner and walk him straight past every turn.
        while self.route and self.route[0] == tile:
            self.route.pop(0)
        if not self.route:
            self.route = self.bfs(tile, goal_tile, PLAN_FACING)
            if not self.route:
                self.stuck += 1
                if self.stuck > 20:
                    self._next_goal(step)
                self.press(step, None)
                return
        nxt = self.route[0]
        d = (nxt[0] - tile[0], nxt[1] - tile[1])
        key = self.DIRKEY.get(d)
        if key is None:
            self.route = []
            self.press(step, None)
            self.cooldown = 2
            return
        if len(self.route) == 1:
            # last hop: a *timed* press, so the key is released before the
            # player lands and a held key cannot carry him past the goal
            self.press(step, None)
            self.emit(step, key, hold=10)
            self.cooldown = 20
            return
        self.press(step, key)
        self.cooldown = 1


def fingerprint(client: LocalPygameClient) -> dict:
    import random as _random

    p = local_session.player
    gv = p.game_variables.get_state()
    fp: dict = {
        "states": list(client.active_state_names),
        "map": client.get_map_name(),
        "tile": list(p.tile_pos),
        "pos": [round(float(p.position.x), 9), round(float(p.position.y), 9)],
        "facing": str(p.facing),
        "moving": bool(p.moving),
        "money": int(p.money_controller.money_manager.get_money()),
        "vars": {k: gv[k] for k in WATCHED_VARS if k in gv},
        "nvars": len(gv),
        # hash of *every* game variable, not just the watched ones -- without
        # this the monster instance-id UUID that add_monster writes into
        # `add_monster` (tuxemon/event/actions/add_monster.py:68) is invisible
        # to the digest, and the TUXEMON_DET_UUID bisect knob reads as a
        # no-op when it is not.
        "vars_hash": hashlib.sha1(
            json.dumps(gv, sort_keys=True, default=str).encode()
        ).hexdigest()[:16],
        "npcs": [
            [
                str(n.slug),
                round(float(n.position.x), 9),
                round(float(n.position.y), 9),
                str(n.facing),
            ]
            for n in sorted(
                client.npc_manager.get_all_entities(), key=lambda n: str(n.slug)
            )
        ],
    }
    try:
        fp["party"] = [
            [str(m.slug), int(m.level), int(m.current_hp), int(m.hp)]
            for m in p.monsters
        ]
    except Exception:
        fp["party"] = None
    fp["rng"] = hashlib.sha1(repr(_random.getstate()).encode()).hexdigest()[:16]
    return fp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scenario", choices=sorted(SCENARIOS), default="intro"
    )
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--dump-tape", action="store_true",
                    help="include the raw keypress tape in --out (needed to seed a TAS loop)")
    ap.add_argument(
        "--samples",
        action="store_true",
        help="also write the full per-step sample list (large)",
    )
    ap.add_argument(
        "--wild",
        choices=("run", "fight"),
        default=None,
        help="wild-encounter policy: flee (default) or fight for levels",
    )
    ap.add_argument(
        "--target-map",
        default=None,
        help="map whose first arrival step is reported (default: the last "
        "map named by the scenario route)",
    )
    ap.add_argument(
        "--stop-on-arrival",
        action="store_true",
        help="stop the run as soon as --target-map is entered (measures the "
        "arrival step only; changes the digest, so do not mix with "
        "determinism comparisons)",
    )
    args = ap.parse_args()

    route, default_steps, teleport, default_wild = SCENARIOS[args.scenario]
    steps = args.steps or default_steps
    wild = args.wild or default_wild
    target_map = args.target_map
    if target_map is None:
        named = [e[0] for e in route if e[0].endswith(".tmx")]
        target_map = named[-1] if named else None

    client = make_client()
    drv = Driver(client, route, wild=wild)

    samples = []
    teleported = False
    prev_map: str | None = None
    prev_goal = 0
    prev_combat = False
    arrival: int | None = None
    for step in range(steps):
        if teleport is not None and not teleported:
            # run the real intro first, then jump (gym only)
            if step >= 2600:
                client.event_engine.execute_action(
                    "teleport", ["player", teleport[0], teleport[1], teleport[2]]
                )
                teleported = True
        drv.step(step)
        client.update(FIXED_DT)
        determinism.advance(FIXED_DT)
        fp = fingerprint(client)
        samples.append([step, fp])

        # -- segment instrumentation (observation only; it never feeds back
        #    into the driver, so it cannot perturb the run) -----------------
        if fp["map"] != prev_map:
            drv.events.append(
                {"step": step, "kind": "map", "value": fp["map"]}
            )
            prev_map = fp["map"]
            if arrival is None and fp["map"] == target_map:
                arrival = step
                drv.events.append(
                    {"step": step, "kind": "arrival", "value": target_map}
                )
                if args.stop_on_arrival:
                    break
        if drv.goal_i != prev_goal:
            drv.events.append(
                {
                    "step": step,
                    "kind": "goal",
                    "value": f"{prev_goal}->{drv.goal_i} "
                    f"{route[prev_goal] if prev_goal < len(route) else 'end'}",
                }
            )
            prev_goal = drv.goal_i
        in_combat = "CombatState" in fp["states"]
        if in_combat != prev_combat:
            if in_combat:
                try:
                    is_wild = not client.combat_session.is_trainer_battle
                except Exception:
                    is_wild = None
                drv.events.append(
                    {
                        "step": step,
                        "kind": "combat_start",
                        "value": (
                            "wild"
                            if is_wild
                            else ("trainer" if is_wild is False else "unknown")
                        ),
                        "map": fp["map"],
                    }
                )
            else:
                drv.events.append(
                    {
                        "step": step,
                        "kind": "combat_end",
                        "value": str(fp["vars"].get("battle_last_result")),
                    }
                )
            prev_combat = in_combat

    blob = json.dumps(samples, sort_keys=True)
    digest = hashlib.sha256(blob.encode()).hexdigest()[:16]
    tape_digest = hashlib.sha256(
        json.dumps(drv.tape, sort_keys=True).encode()
    ).hexdigest()[:16]

    maps = [s[1]["map"] for s in samples]
    tiles = {(s[1]["map"], tuple(s[1]["tile"])) for s in samples}
    seen_vars: dict[str, set] = {}
    for s in samples:
        for k, v in s[1]["vars"].items():
            seen_vars.setdefault(k, set()).add(str(v))

    result = {
        "scenario": args.scenario,
        "wild_policy": wild,
        "digest": digest,
        "tape_digest": tape_digest,
        "tape_events": len(drv.tape),
        "steps_sampled": len(samples),
        "deterministic_mode": determinism.is_enabled(),
        "seed": determinism.seed(),
        "teleported": teleport is not None,
        "distinct_maps": sorted(set(maps)),
        "distinct_map_tiles": len(tiles),
        "distinct_states": sorted({n for s in samples for n in s[1]["states"]}),
        "distinct_party_states": len(
            {json.dumps(s[1]["party"], sort_keys=True) for s in samples}
        ),
        "distinct_rng_states": len({s[1]["rng"] for s in samples}),
        "distinct_var_states": len({s[1]["vars_hash"] for s in samples}),
        "watched_vars": {k: sorted(v) for k, v in sorted(seen_vars.items())},
        "final_map": samples[-1][1]["map"],
        "final_tile": samples[-1][1]["tile"],
        "final_party": samples[-1][1]["party"],
        "final_rng": samples[-1][1]["rng"],
        "final_states": samples[-1][1]["states"],
        "target_map": target_map,
        "arrival_step": arrival,
        "reached_target": arrival is not None,
        "wild_encounters": sum(
            1
            for e in drv.events
            if e["kind"] == "combat_start" and e["value"] == "wild"
        ),
        "trainer_battles": sum(
            1
            for e in drv.events
            if e["kind"] == "combat_start" and e["value"] == "trainer"
        ),
        "rewinds": sum(1 for e in drv.events if e["kind"] == "rewind"),
        "map_order": [e["value"] for e in drv.events if e["kind"] == "map"],
        "events": drv.events,
    }
    payload: dict = {"summary": result}
    if args.samples:
        payload["samples"] = samples
    # The raw keypress tape. Only its DIGEST and LENGTH were reported, which is enough to prove
    # two runs matched but not enough to reuse the run: an optimisation loop needs the actual
    # inputs to seed from, and re-deriving them from the route would mean re-implementing the
    # driver's pathfinding. Each entry is (step, "down"|"up", pygame key code).
    if args.dump_tape:
        payload["tape"] = drv.tape
    Path(args.out).write_text(json.dumps(payload, sort_keys=True))
    terse = {k: v for k, v in result.items() if k != "events"}
    print("RESULT " + json.dumps(terse, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
