#!/usr/bin/env python3
"""Agent-playable Tuxemon environment: step(action) -> (screen, state, done).

WHAT THIS IS FOR
det_probe_spyder.py drives a FIXED, scripted route and reports a digest -- it measures whether the
engine is deterministic, not whether a player can play. This wraps the same machinery so an
external agent supplies the inputs instead of a hardcoded tape, which is what a speedrun benchmark
arm needs.

IT WRAPS det_probe_spyder, IT DOES NOT REIMPLEMENT IT. The boot sequence, the synthetic-keyboard
shim, the fingerprint and the menu policy all come from there by import. A second copy of the
driver loop would drift from the measured baseline, and the baseline is the only thing that makes
a score meaningful.

DETERMINISM IS NOT AUTOMATIC -- SET TUXEMON_DETERMINISTIC=1.
Without it the engine runs on wall-clock and two identical action sequences produce different
outcomes, so a score is not reproducible and A/B comparisons between models are meaningless. This
module REFUSES to construct unless the flag is on, because the failure is silent otherwise:
det_probe reports `deterministic_mode` in its output and it reads `false`, while digests differ
run to run in a way that looks exactly like a real engine bug. (That misreading cost a wrong
"screen resolution changes the run" conclusion during development; resolution in fact changes
nothing -- verified identical digests at 256x144 and 1280x720.)

THE INTRO IS NOT AGENT WORK. Character creation is a chain of pygame_menu dialogs (scenario,
gender, pronoun, race, name, starter monster) that the probe's Driver auto-answers. An agent that
had to navigate those would spend its budget on a fixed cutscene that has exactly one sensible
path, and every arm would pay the same toll. reset() therefore runs the intro under the Driver's
own policy and hands over at the first frame the player controls the world -- WorldState alone,
in the bedroom, not moving. `intro_steps` in the returned info records what that cost, so it can
be subtracted or reported.

COMBAT IS AUTO-RESOLVED BY DEFAULT (auto_combat=True). The route to the gym contains a MANDATORY
trainer battle (Billie, on spyder_route2) with no Run option, and losing it warps the player back
to the bedroom. Battle tactics are a different skill from navigation, and the baseline was
measured with the Driver's combat policy, so leaving combat to the agent would make agent runs
non-comparable to it. Pass auto_combat=False to give the agent the battle menus too -- then do not
compare against the 120k-step baseline.

USAGE
    TUXEMON_DETERMINISTIC=1 python3 -c "
    import sys; sys.path.insert(0, 'tools')
    from tux_env import TuxemonEnv
    env = TuxemonEnv()
    obs, info = env.reset()
    for a in ['DOWN']*20:
        obs, st, done, info = env.step(a)
    print(env.state()['map'], env.state()['tile'])"

The default goal is spyder_leather_gym.tmx, the same target the gymwalk baseline measures.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
# Frames are only produced when drawing is on; an agent that gets images needs this.
os.environ.setdefault("TUXEMON_DETERMINISTIC_DRAW", "1")
# Rendering at 1280x720 makes scale=5 while tiles blit at native 16px, so a captured frame is a
# lattice of tiles on black (3.4% non-black). Native resolution keeps pitch == art size, and the
# digest is identical either way, so this is free.
os.environ.setdefault("TUXEMON_NATIVE_RENDER", "1")

import pygame  # noqa: E402

import det_probe_spyder as P  # noqa: E402
from tuxemon import determinism  # noqa: E402

# Agent-facing button names -> the keys the shim posts. Deliberately small: this is the whole
# controller. RETURN is confirm/interact and ESCAPE is cancel/back, matching Tuxemon's own
# bindings, so an agent prompt can describe them without naming pygame internals.
ACTIONS: dict[str, int | None] = {
    "UP": pygame.K_UP,
    "DOWN": pygame.K_DOWN,
    "LEFT": pygame.K_LEFT,
    "RIGHT": pygame.K_RIGHT,
    "INTERACT": pygame.K_RETURN,
    "BACK": pygame.K_ESCAPE,
    "NOOP": None,
}

DEFAULT_TARGET = "spyder_leather_gym.tmx"

# ONE ENV PER PROCESS. Tuxemon's client binds a TCP port for its CLI/server and stores the player
# on the module-global `local_session`, so constructing a second env in the same interpreter
# raises OSError 98 (address already in use) and leaves the second client wired to the first
# one's session -- its reset() then hangs on ImageState until intro_cap and reports a timeout
# that looks like an engine bug. Fail loudly here instead: run each arm in its own process.
_ENV_CREATED = False


class TuxemonEnv:
    """One Tuxemon run under agent control.

    step() returns (observation, state, done, info). `done` is True once the target map is
    entered or max_steps is exhausted; info["reached_target"] distinguishes the two.
    """

    def __init__(
        self,
        target_map: str = DEFAULT_TARGET,
        auto_combat: bool = True,
        max_steps: int = 200_000,
        wild: str = "fight",
        handoff_map: str = "spyder_bedroom.tmx",
    ) -> None:
        global _ENV_CREATED
        if _ENV_CREATED:
            raise RuntimeError(
                "a TuxemonEnv already exists in this process. The client binds a TCP port and "
                "uses a global session, so a second one collides (OSError 98) and its reset() "
                "hangs. Run each arm in its own process."
            )
        if not determinism.is_enabled():
            raise RuntimeError(
                "TUXEMON_DETERMINISTIC=1 is required. Without it the engine runs on wall-clock "
                "time, identical action sequences diverge, and no score is reproducible. NOTE "
                "TUXEMON_DETERMINISTIC_DRAW=1 is a DIFFERENT flag that only forces drawing -- "
                "setting it alone leaves determinism off and the breakage is silent."
            )
        self.target_map = target_map
        self.auto_combat = auto_combat
        self.max_steps = max_steps
        self.wild = wild
        self.handoff_map = handoff_map

        self.client = None
        self.drv = None
        self.step_count = 0        # engine ticks consumed, INCLUDING the intro
        self.agent_steps = 0       # step() calls made by the agent
        self.intro_steps = 0
        self.reached_target = False
        self.maps_seen: list[str] = []
        self._held: int | None = None
        _ENV_CREATED = True

    # -- engine ------------------------------------------------------------
    def _tick(self, drive: bool) -> None:
        """One fixed-dt engine tick.

        `drive` runs the probe's Driver policy for this tick (used for the intro and, when
        auto_combat is on, for battles). The route is empty, so the Driver only ever answers
        menus -- it never navigates the world on the agent's behalf.
        """
        if drive:
            self.drv.step(self.step_count)
        self.client.update(P.FIXED_DT)
        determinism.advance(P.FIXED_DT)
        self.step_count += 1

    def _in_menu_or_combat(self) -> bool:
        nm, _ = self.drv.top_menu()
        return nm is not None

    def _world_ready(self) -> bool:
        """Player controls the world: no menu/wait state on top, and standing still.

        Do NOT test `active_state_names == ["WorldState"]`. BackgroundState is a persistent
        backdrop that sits in the stack for the whole run, so the exact-match form never becomes
        true and reset() times out after the intro has in fact completed. Ask the Driver's own
        top_menu() instead -- it walks the stack, ignores non-menu states, and returns None once
        WorldState is reachable, which is exactly the condition "the player has control".
        """
        names = list(self.client.active_state_names)
        if "WorldState" not in names:
            return False
        # Overlay states that are neither menus nor WAIT_STATES, so top_menu() reports "clear"
        # while they own the screen. ImageState is the intro splash: without this the handoff
        # fires at tick 68 on start_tuxemon.tmx with an EMPTY PARTY -- before the starter monster
        # is chosen -- and the agent inherits a half-built save.
        if any(n in ("ImageState", "SinkState", "TeleporterState") for n in names):
            return False
        nm, _ = self.drv.top_menu()
        if nm is not None:
            return False
        try:
            player = P.local_session.player
        except Exception:
            return False
        # DO NOT gate on a non-empty party. The starter monster is NOT granted during character
        # creation -- traced tick-by-tick, the party is empty at handoff and stays empty until the
        # player walks to the event that awards it, well into the route. Requiring monsters here
        # makes reset() time out after 20k ticks on a run that is in fact ready at ~2500.
        #
        # The map test is what actually marks the end of the intro: the sequence runs
        # start_tuxemon -> spyder_bedroom -> spyder_paper_scoop -> back to spyder_bedroom, and
        # only the final arrival leaves a clean stack with the player idle and controllable.
        if self.client.get_map_name() != self.handoff_map:
            return False
        return not bool(player.moving)

    # -- api ---------------------------------------------------------------
    def reset(self, intro_cap: int = 20_000):
        """Boot, auto-play character creation, hand over at first player control."""
        self.client = P.make_client()
        # Empty route: the Driver answers menus but issues no navigation of its own.
        self.drv = P.Driver(self.client, [], wild=self.wild)
        self.step_count = self.agent_steps = 0
        self.reached_target = False
        self.maps_seen = []

        for _ in range(intro_cap):
            self._tick(drive=True)
            if self._world_ready():
                break
        else:
            raise RuntimeError(
                f"character creation did not finish within {intro_cap} ticks; "
                f"states={list(self.client.active_state_names)}"
            )
        self.intro_steps = self.step_count
        st = self.state()
        self.maps_seen.append(st["map"])
        return self.observation(), {"intro_steps": self.intro_steps, "state": st}

    def step(self, action: str, hold: int = 4, frames: int = 8):
        """Apply one action.

        `hold` ticks with the key down, then release and run to `frames` total. The defaults match
        the probe's own cadence (hold=4), so agent input lands the same way the measured tape's
        input did -- a shorter hold silently drops movement inputs the engine only latches after a
        few frames.
        """
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}; valid: {sorted(ACTIONS)}")
        if self.client is None:
            raise RuntimeError("call reset() first")

        key = ACTIONS[action]
        busy = self.auto_combat and self._in_menu_or_combat()

        if busy:
            # A menu or battle owns the screen. The agent's action is not applied: the Driver
            # resolves it instead. Reported in info["agent_control"]=False so a caller can tell
            # "my action did nothing" from "my action did nothing USEFUL".
            for _ in range(frames):
                self._tick(drive=True)
                if not self._in_menu_or_combat():
                    break
        else:
            if key is not None:
                P.post_key(pygame.KEYDOWN, key)
                self._held = key
            for i in range(frames):
                if key is not None and i == hold:
                    P.post_key(pygame.KEYUP, key)
                    self._held = None
                self._tick(drive=False)
            if self._held is not None:      # frames <= hold: release anyway, never leak a hold
                P.post_key(pygame.KEYUP, self._held)
                self._held = None

        self.agent_steps += 1
        st = self.state()
        if st["map"] and (not self.maps_seen or st["map"] != self.maps_seen[-1]):
            self.maps_seen.append(st["map"])
        if st["map"] == self.target_map:
            self.reached_target = True

        done = self.reached_target or self.step_count >= self.max_steps
        info = {
            "agent_control": not busy,
            "reached_target": self.reached_target,
            "engine_steps": self.step_count,
            "agent_steps": self.agent_steps,
            "maps_seen": list(self.maps_seen),
        }
        return self.observation(), st, done, info

    def state(self) -> dict:
        """Ground-truth state. A subset of the probe's fingerprint -- the digest-only fields
        (rng, vars_hash, npc table) are omitted because handing an agent the RNG state would let
        it predict encounters the player cannot see."""
        fp = P.fingerprint(self.client)
        return {
            "map": fp["map"],
            "tile": fp["tile"],
            "facing": fp["facing"],
            "moving": fp["moving"],
            "money": fp["money"],
            "party": fp["party"],
            "states": fp["states"],
        }

    def observation(self):
        """Current frame as a PIL image, or None if drawing is off."""
        surf = getattr(self.client, "screen", None)
        if surf is None:
            return None
        from PIL import Image

        return Image.frombytes("RGB", surf.get_size(), pygame.image.tostring(surf, "RGB"))

    def score(self) -> dict:
        """Benchmark result. Lower engine_steps is better; a run that never reaches the target
        scores no steps at all rather than a large number, so partial progress is reported by
        maps_reached instead of being folded into a single misleading figure."""
        return {
            "reached_target": self.reached_target,
            "target_map": self.target_map,
            "engine_steps": self.step_count if self.reached_target else None,
            "agent_steps": self.agent_steps if self.reached_target else None,
            "intro_steps": self.intro_steps,
            "maps_reached": len(set(self.maps_seen)),
            "maps_seen": self.maps_seen,
            "final_map": self.state()["map"] if self.client else None,
        }


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="smoke-test the env with a scripted action list")
    ap.add_argument("--actions", default="DOWN:40,LEFT:10,INTERACT:2",
                    help="comma-separated ACTION:count")
    ap.add_argument("--shot", default=None, help="write the final frame here")
    a = ap.parse_args()

    env = TuxemonEnv()
    _, info = env.reset()
    print(f"intro finished in {info['intro_steps']} ticks; state={info['state']}")
    for part in a.actions.split(","):
        name, _, n = part.partition(":")
        for _ in range(int(n or 1)):
            _, st, done, inf = env.step(name.strip().upper())
            if done:
                break
    print(json.dumps(env.score(), indent=1))
    if a.shot:
        img = env.observation()
        if img:
            img.save(a.shot)
            print("wrote", a.shot)
