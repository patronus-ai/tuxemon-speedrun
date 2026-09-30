#!/usr/bin/env python3
"""Convert the gymwalk driver tape into the env's action space, CLIPPED to the scoring window.

WHY THIS EXISTS. The previous converter (tux_seed_from_tape.py) ran 85 ticks long and never
reproduced the baseline. The cause was not the hold model, not the action format, and not RNG:
the driver plays on to its 120,000-step budget, so the tape carries 6 events firing up to 100
frames PAST the gym arrival at 105,397. Those trailing frames were converted along with the rest.
In a deterministic engine an 85-tick surplus is fatal on its own.

The tape itself is exactly expressible: over 105,496 frames it has 1,453 downs / 1,453 ups, never
two keys down at once, and zero overlapping frames. So the conversion is a windowing problem.

The window is [HANDOFF, ARRIVAL]. Below HANDOFF the env replays the driver's own intro route, so
those events are already applied; above ARRIVAL the run is over and the input is noise.

Emit `frames` = this press's span (down -> next down, clipped at ARRIVAL) and `hold` = down -> its
own up. With no overlaps span >= hold always, so the key is down for `hold` ticks and idle for the
remainder -- which is exactly what the driver did. Replay with auto_combat=False so the env does
not substitute its own battle inputs for the tape's.
"""
import argparse
import json
import sys

# pygame keycodes -> tux_env ACTIONS. These are pygame 2 / SDL2 values (the arrows are in the
# 0x40000xx scancode range), NOT the pygame 1.x 273-276 block: with the old constants every arrow
# press converts to NOOP and the tape walks nowhere while still summing to the right tick count.
KEYMAP = {1073741906: "UP", 1073741905: "DOWN", 1073741904: "LEFT", 1073741903: "RIGHT",
          13: "INTERACT", 27: "BACK"}


def build(tape, handoff, arrival):
    ev = sorted(((int(s), str(t), int(k)) for s, t, k in tape), key=lambda x: (x[0], x[1] != "up"))
    downs = [(s, k) for s, t, k in ev if t == "down" and handoff <= s < arrival]
    ups = [(s, k) for s, t, k in ev if t == "up"]

    out = []
    # Gap between the handoff and the first press: real idle time, not something to skip.
    if downs and downs[0][0] > handoff:
        out.append(["NOOP", downs[0][0] - handoff, downs[0][0] - handoff, 0])

    for i, (s, k) in enumerate(downs):
        act = KEYMAP.get(k)
        if act is None:
            print(f"  WARN: unmapped keycode {k} at step {s}; emitting NOOP", file=sys.stderr)
            act = "NOOP"
        up = next((u for u, uk in ups if uk == k and u > s), arrival)
        nxt = downs[i + 1][0] if i + 1 < len(downs) else arrival
        # Clip BOTH at the goal: the whole point of this converter.
        up, nxt = min(up, arrival), min(nxt, arrival)
        hold = max(0, up - s)
        frames = max(hold, nxt - s)
        if frames > 0:
            out.append([act, frames, hold])
    return out


def rle(actions):
    """[ACTION, FRAMES, HOLD] -> [ACTION, COUNT, FRAMES, HOLD], collapsing identical neighbours."""
    packed = []
    for a in actions:
        if len(a) == 4:  # NOOP rows already carry a count slot
            a = [a[0], a[1], a[2]]
        if packed and packed[-1][0] == a[0] and packed[-1][2] == a[1] and packed[-1][3] == a[2]:
            packed[-1][1] += 1
        else:
            packed.append([a[0], 1, a[1], a[2]])
    return packed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="det_probe_spyder --dump-tape output")
    ap.add_argument("--handoff", type=int, default=2291)
    ap.add_argument("--arrival", type=int, default=None, help="default: arrival_step from the dump")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d = json.load(open(a.dump))
    # arrival_step lives under "summary" in the --dump-tape payload, not at the top level.
    summary = d.get("summary") or {}
    arrival = a.arrival or int(summary.get("arrival_step") or d["arrival_step"])
    acts = build(d["tape"], a.handoff, arrival)
    packed = rle(acts)

    ticks = sum(c * f for _, c, f, _ in packed)
    want = arrival - a.handoff
    json.dump(packed, open(a.out, "w"))
    print(f"  entries={len(packed)} actions={sum(c for _, c, _, _ in packed)}")
    print(f"  ticks={ticks} want={want} delta={ticks - want:+d} -> {a.out}")
    # A tick-exact tape is the whole premise; refuse to look successful otherwise.
    return 0 if ticks == want else 1


if __name__ == "__main__":
    sys.exit(main())
