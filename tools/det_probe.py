#!/usr/bin/env python3
"""
Determinism probe for Tuxemon.

Boots the game headless (SDL dummy video), drives it a fixed number of
fixed-dt steps, injects a scripted input tape at exact step numbers, and
prints a per-step state fingerprint plus a final digest.

The step count is decided by the harness, never by the wall clock, so a
digest mismatch is engine divergence and not observer jitter.

Usage:
    python tools/det_probe.py --steps 1200 --out run1.json
    TUXEMON_DETERMINISTIC=1 python tools/det_probe.py --steps 1200 --out run1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
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
from tuxemon.prepare import pygame_init  # noqa: E402
from tuxemon.session import local_session  # noqa: E402
from tuxemon.startup_state_machine import StartupStateMachine  # noqa: E402
from tuxemon.user_config import CONFIG  # noqa: E402

FIXED_DT = 1.0 / 60.0

START_MAP = "cotton_town.tmx"
START_TILE = (12, 12)
WARMUP_STEPS = 120

# route1.tmx has a "random_encounter default_encounter" event object at
# x=624 y=464 w=160 h=32 (16 px tiles) -> tiles x 39..48, y 29..30.  It is
# gated on the `badge` variable, which the probe sets directly.
ENCOUNTER_MAP = "route1.tmx"
ENCOUNTER_TILE = (40, 29)

# (key, down_step, up_step) -- steps are relative to the end of warmup
DEFAULT_TAPE = [
    (pygame.K_DOWN, 10, 55),
    (pygame.K_LEFT, 70, 120),
    (pygame.K_UP, 140, 190),
    (pygame.K_RIGHT, 210, 260),
    (pygame.K_DOWN, 280, 340),
    (pygame.K_RIGHT, 360, 420),
    (pygame.K_UP, 440, 500),
    (pygame.K_LEFT, 520, 590),
    (pygame.K_DOWN, 610, 680),
    (pygame.K_RIGHT, 700, 780),
    (pygame.K_UP, 800, 870),
    (pygame.K_LEFT, 890, 960),
    (pygame.K_DOWN, 980, 1060),
]


def build_tape() -> dict[int, list[tuple[int, int]]]:
    tape: dict[int, list[tuple[int, int]]] = {}
    for key, down, up in DEFAULT_TAPE:
        tape.setdefault(down, []).append((pygame.KEYDOWN, key))
        tape.setdefault(up, []).append((pygame.KEYUP, key))
    return tape


def make_client() -> LocalPygameClient:
    context = pygame_init()
    config = CONFIG.copy()
    # straight-to-world boot: no title screen, no splash, no CLI thread
    config.config_model.game.skip_titlescreen = True
    config.config_model.display.splash = False
    config.config_model.game.cli_enabled = False
    config.mods = ["tuxemon"]
    client = LocalPygameClient.create(config, context)
    local_session.set_client(client)
    StartupStateMachine(client, config).run()

    # drop the intro cutscene stack and start from a plain free-roam map
    for name in ("ChoiceState", "ImageState", "DialogState", "BackgroundState"):
        try:
            client.remove_state_by_name(name)
        except Exception:
            pass
    client.event_engine.execute_action(
        "teleport", ["player", START_MAP, START_TILE[0], START_TILE[1]]
    )
    return client


def fingerprint(client: LocalPygameClient) -> dict:
    import random as _random

    fp: dict = {}
    fp["states"] = list(client.active_state_names)
    fp["map"] = client.get_map_name()
    p = local_session.player
    fp["pos"] = [
        round(float(p.position.x), 9),
        round(float(p.position.y), 9),
    ]
    fp["tile"] = list(p.tile_pos)
    fp["facing"] = str(p.facing)
    fp["moving"] = bool(p.moving)
    fp["money"] = int(p.money_controller.money_manager.get_money())
    fp["npcs"] = [
        [
            str(n.slug),
            round(float(n.position.x), 9),
            round(float(n.position.y), 9),
            str(n.facing),
        ]
        for n in sorted(
            client.npc_manager.get_all_entities(), key=lambda n: str(n.slug)
        )
    ]
    try:
        fp["party"] = [
            [str(m.slug), int(m.level), int(m.current_hp), int(m.hp)]
            for m in p.monsters
        ]
    except Exception:
        fp["party"] = None
    fp["rng"] = hashlib.sha1(repr(_random.getstate()).encode()).hexdigest()[
        :16
    ]
    return fp


def build_battle_tape() -> dict[int, list[tuple[int, int]]]:
    """
    Pace back and forth inside a wild-encounter zone while mashing CONFIRM.

    The walking triggers `random_encounter` through the engine's own
    (non-blocking) event path; the CONFIRM presses drive the combat menus.
    This is by far the most RNG-heavy code path in the game.
    """
    tape: dict[int, list[tuple[int, int]]] = {}
    key = pygame.K_RIGHT
    for step in range(0, 6000, 60):
        tape.setdefault(step, []).append((pygame.KEYDOWN, key))
        tape.setdefault(step + 40, []).append((pygame.KEYUP, key))
        key = pygame.K_LEFT if key == pygame.K_RIGHT else pygame.K_RIGHT
    for step in range(10, 6000, 16):
        tape.setdefault(step, []).append((pygame.KEYDOWN, pygame.K_RETURN))
        tape.setdefault(step + 3, []).append((pygame.KEYUP, pygame.K_RETURN))
    return tape


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument(
        "--scenario", choices=("walk", "battle"), default="walk"
    )
    args = ap.parse_args()

    tape = build_tape() if args.scenario == "walk" else build_battle_tape()
    client = make_client()

    if args.scenario == "battle":
        client.event_engine.execute_action("set_environment", ("grass",))
        for slug, lvl in (("rockitten", 18), ("budaye", 17)):
            client.event_engine.execute_action("add_monster", (slug, lvl))
        client.event_engine.execute_action(
            "teleport",
            ["player", ENCOUNTER_MAP, ENCOUNTER_TILE[0], ENCOUNTER_TILE[1]],
        )

    for _ in range(WARMUP_STEPS):
        client.update(FIXED_DT)
        determinism.advance(FIXED_DT)

    if args.scenario == "battle":
        # skip=True runs start() (which pushes CombatState) without entering
        # EventAction.run()'s blocking wall-clock loop.
        client.event_engine.execute_action(
            "wild_encounter", ("bamboon", 16), skip=True
        )

    samples = []
    for step in range(args.steps):
        for ev_type, key in tape.get(step, []):
            pygame.event.post(
                pygame.event.Event(
                    ev_type, key=key, mod=0, unicode="", scancode=0
                )
            )
        client.update(FIXED_DT)
        determinism.advance(FIXED_DT)
        samples.append([step, fingerprint(client)])

    blob = json.dumps(samples, sort_keys=True)
    digest = hashlib.sha256(blob.encode()).hexdigest()[:16]

    tiles = {tuple(s[1]["tile"]) for s in samples}
    npc_hashes = {json.dumps(s[1]["npcs"], sort_keys=True) for s in samples}
    result = {
        "digest": digest,
        "steps_sampled": len(samples),
        "distinct_tiles": len(tiles),
        "distinct_npc_states": len(npc_hashes),
        "final_tile": samples[-1][1]["tile"] if samples else None,
        "final_rng": samples[-1][1]["rng"] if samples else None,
        "final_map": samples[-1][1]["map"] if samples else None,
        "final_states": samples[-1][1]["states"] if samples else None,
        "states_seen": sorted(
            {n for s in samples for n in s[1]["states"]}
        ),
        "distinct_party_states": len(
            {json.dumps(s[1]["party"], sort_keys=True) for s in samples}
        ),
        "final_party": samples[-1][1]["party"] if samples else None,
        "deterministic_mode": determinism.is_enabled(),
    }
    Path(args.out).write_text(
        json.dumps({"summary": result, "samples": samples}, sort_keys=True)
    )
    print("RESULT " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
