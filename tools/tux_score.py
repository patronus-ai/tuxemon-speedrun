#!/usr/bin/env python3
"""Score a Tuxemon action tape: replay it through TuxemonEnv and report progress toward the gym.

OBJECTIVE. Fewest ENGINE STEPS to enter spyder_leather_gym.tmx. The measured baseline reaches it
at step 105,397 of a 120,000-step scenario, so there is a lot of headroom -- but the headroom is
not evenly spread. That run loses the mandatory Billie battle on spyder_route2 FIVE times, and
each loss warps the player back to spyder_bedroom to walk the whole route again (visible as five
repeats of route1 -> cotton_town -> route2 in its map_order, and as "rewinds": 5). Most of the
105k steps are those retries, not travel. An optimiser that wins that battle earlier removes a
large, genuinely compressible chunk; this is a real speedrun target, not a metric artifact.

A RUN THAT NEVER ARRIVES SCORES NO STEPS AT ALL. Reporting a large step count for a failure would
make "wandered for 120k steps" look like a slow success and let a search hill-climb on distance
travelled. Failures rank by how much of the route they covered (maps_reached) instead, and always
below any arrival.

MUST RUN AS ITS OWN PROCESS. TuxemonEnv binds a TCP port and uses a module-global session, so a
second env in the same interpreter raises OSError 98 and silently wires itself to the first one's
game -- its reset() then hangs and reports a timeout that reads like an engine fault. The loop
invokes this file as a subprocess for exactly that reason; do not import and call it in-process.

TUXEMON_DETERMINISTIC=1 IS REQUIRED and TuxemonEnv refuses without it. Note the near-miss
TUXEMON_DETERMINISTIC_DRAW=1, which only forces drawing: setting that alone leaves determinism off
and identical tapes then score differently, which looks exactly like a real engine bug.

    TUXEMON_DETERMINISTIC=1 python3 tux_score.py --tape tape.json --out score.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

# tux_env lives beside this file in tools/; resolve it from here rather than a host path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

VALID = {"UP", "DOWN", "LEFT", "RIGHT", "INTERACT", "BACK", "NOOP"}


def expand(rle, default_hold=4, default_frames=8):
    """[[ACTION, COUNT]] or [[ACTION, COUNT, FRAMES, HOLD]] -> [(action, frames, hold), ...].

    THE OPTIONAL TIMING IS WHAT MAKES A DRIVER-DERIVED SEED FAITHFUL. With a single fixed cadence
    for every action, a converted baseline tape drifts: the driver pressed at arbitrary step
    numbers with variable holds, so rounding every press to one fixed-length slot accumulates
    error until the geometry no longer matches. Measured: the baseline reaches the gym, but its
    fixed-cadence conversion stalled at spyder_paper_town with 5 of 12 maps -- a walk that needed
    4 tiles took 3, and everything after aimed at the wrong tile.

    2-element entries keep the old meaning (agent-authored tapes are unaffected); 4-element
    entries pin FRAMES and HOLD per action so a recorded tape replays at its original cadence.
    """
    out = []
    n4 = sum(1 for e in rle if isinstance(e, (list, tuple)) and len(e) == 4)
    expand.last_all_2el = bool(rle) and n4 == 0   # read by main() for the seeded-run guard
    for ent in rle:
        if not isinstance(ent, (list, tuple)) or len(ent) not in (2, 4):
            raise ValueError(f"bad entry {ent!r}: want [ACTION, COUNT] or [ACTION, COUNT, FRAMES, HOLD]")
        act, n = str(ent[0]).upper(), int(ent[1])
        frames = int(ent[2]) if len(ent) == 4 else default_frames
        hold = int(ent[3]) if len(ent) == 4 else default_hold
        if act not in VALID:
            raise ValueError(f"unknown action {act!r}; valid: {sorted(VALID)}")
        if n < 0:
            raise ValueError(f"negative count in {ent!r}")
        if frames < 1 or hold < 0 or hold > frames:
            raise ValueError(f"bad timing in {ent!r}: need 1<=frames, 0<=hold<=frames")
        out.extend([(act, frames, hold)] * n)
    return out


def rank(r: dict) -> tuple:
    """Best first. Arrival always beats non-arrival; then fewer steps; then more of the route."""
    return (0 if r["reached_target"] else 1,
            r["engine_steps"] if r["reached_target"] else 10 ** 9,
            -r["maps_reached"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tape", required=True, help="JSON list of [ACTION, COUNT]")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-steps", type=int, default=200_000)
    ap.add_argument("--target", default="spyder_leather_gym.tmx")
    ap.add_argument("--trail", default=None, help="write the per-action map trail here")
    # CADENCE MUST MATCH THE TAPE IT SCORES. TuxemonEnv.step defaults to hold=4/frames=8, but the
    # baseline driver presses with hold=4 and a down-to-down gap of 11 (measured: 69 of 90 gaps
    # are exactly 11). Scoring a driver-derived seed at frames=8 lands every action 3 ticks early
    # and the drift accumulates, so the seed does not reproduce and reads as a bad tape.
    ap.add_argument("--hold", type=int, default=4)
    ap.add_argument("--frames", type=int, default=8,
                    help="ticks per action; use 11 for driver-derived seed tapes")
    # SEEDED SCORING NEEDS THE SAME INTRO THE SEED WAS CUT FROM. With no route, reset() only
    # answers menus and hands over in the bedroom, while the tape was recorded by a driver that
    # had already walked on -- at step 2291 the driver is in spyder_paper_scoop and the env is in
    # spyder_bedroom, so the seed's inputs address the wrong position and it never leaves the
    # first room. Replaying the same route makes both trajectories identical up to the handoff.
    ap.add_argument("--intro-route", default=None,
                    help="SCENARIOS key (e.g. gymwalk) to replay during reset; omit for scratch")
    ap.add_argument("--handoff-step", type=int, default=None,
                    help="hand over at this engine step; must match the seed's --skip-before")
    # A DRIVER-DERIVED SEED CANNOT BE SCORED WITH auto_combat=True. When combat is auto-resolved
    # the env DISCARDS the tape's input for the duration of the battle and lets its own Driver
    # play instead, and its busy loop breaks out early once the battle ends -- so it consumes a
    # different number of ticks than the recording did and every later input lands on the wrong
    # frame. The gymwalk baseline is wild_policy="fight" with 8 trainer battles and 15 wild
    # encounters, so this is not an edge case: the same tape reaches 11 maps with auto_combat off
    # and 4 with it on. Default stays True so the running from-scratch fleet is unaffected;
    # seeded scoring must pass --no-auto-combat.
    ap.add_argument("--auto-combat", dest="auto_combat", action="store_true", default=True,
                    help="auto-resolve battles (default; agent does navigation only)")
    ap.add_argument("--no-auto-combat", dest="auto_combat", action="store_false",
                    help="apply the tape's own battle inputs; REQUIRED to replay a driver seed")
    a = ap.parse_args()

    if os.environ.get("TUXEMON_DETERMINISTIC") != "1":
        sys.exit("TUXEMON_DETERMINISTIC=1 is required (TUXEMON_DETERMINISTIC_DRAW is NOT it)")

    actions = expand(json.loads(Path(a.tape).read_text()),
                     default_hold=a.hold, default_frames=a.frames)

    # SEEDED RUN + TIMING-LESS TAPE = THE AGENT DID NOT EDIT THE SEED.
    # --intro-route/--handoff-step are passed only by seeded arms. If such a run's candidate is
    # entirely 2-element, the seed's real per-action FRAMES/HOLD were silently replaced by one
    # fixed cadence (measured: 103,206 ticks collapse to 10,952). That is not a bad edit, it is a
    # different tape. It produced 62 consecutive identical failures with NOTHING in any log
    # saying the timings had been substituted, and I read that as proof the task was impossible.
    # Surface it in the result so it shows up on turn 1 instead of turn 64.
    seeded = bool(a.intro_route or a.handoff_step)
    timing_warning = None
    if seeded and getattr(expand, "last_all_2el", False):
        timing_warning = ("candidate is entirely 2-element in a SEEDED run: the seed's per-action "
                          "FRAMES/HOLD were replaced by defaults "
                          f"(hold={a.hold}, frames={a.frames}). Edit the 4-element seed in "
                          "results/speedrun/best/ instead of authoring a new tape.")
        print("WARNING: " + timing_warning, file=sys.stderr)

    from tux_env import TuxemonEnv
    env = TuxemonEnv(target_map=a.target, max_steps=a.max_steps,
                     intro_route=a.intro_route, handoff_step=a.handoff_step,
                     auto_combat=a.auto_combat)
    _, info = env.reset(intro_cap=(a.handoff_step + 500) if a.handoff_step else 20_000)

    trail = []
    prev_map = env.state()["map"]
    for i, (act, frames, hold) in enumerate(actions):
        _, st, done, inf = env.step(act, hold=hold, frames=frames)
        if st["map"] != prev_map:
            trail.append({"action_index": i, "engine_step": env.step_count, "map": st["map"]})
            prev_map = st["map"]
        if done:
            break

    r = env.score()
    r["n_actions_supplied"] = len(actions)
    r["n_actions_used"] = env.agent_steps
    r["cadence"] = {"hold": a.hold, "frames": a.frames}
    # intro_steps is the fixed character-creation cost every candidate pays (~2291 ticks). Report
    # arrival both ways so a comparison against the baseline is not silently off by the intro.
    if timing_warning:
        r["timing_warning"] = timing_warning
    r["steps_after_handoff"] = (env.step_count - env.intro_steps) if r["reached_target"] else None
    r["rank"] = rank(r)
    print(json.dumps(r, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=1))
    if a.trail:
        Path(a.trail).write_text(json.dumps(trail, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
