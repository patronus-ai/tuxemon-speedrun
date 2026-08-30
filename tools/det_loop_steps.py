#!/usr/bin/env python3
"""
Show how many simulation steps Tuxemon's own main loop executes for a fixed
amount of *real* time.

Stock (accumulator) loop: the step count is a function of the wall clock, so
it changes from run to run and collapses under load -> identical inputs land
on different steps -> divergence.

TUXEMON_DETERMINISTIC=1: one step per iteration, so the step count is decided
by TUXEMON_MAX_STEPS / the driver, never by the machine.

Usage:
    python tools/det_loop_steps.py --seconds 3
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from tuxemon import determinism  # noqa: E402

determinism.install()

# real clock, captured before/around the shim
REAL_TIME = determinism._real["time"]
REAL_SLEEP = determinism._real["sleep"]

from tools.det_probe import make_client  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=3.0)
    args = ap.parse_args()

    client = make_client()

    counter = {"n": 0}
    original_update = client.update

    def counting_update(dt: float) -> None:
        counter["n"] += 1
        original_update(dt)

    client.update = counting_update  # type: ignore[method-assign]

    deadline = REAL_TIME() + args.seconds

    def watchdog() -> None:
        while REAL_TIME() < deadline:
            REAL_SLEEP(0.02)
        client.quit()

    t = threading.Thread(target=watchdog, daemon=True)
    t.start()
    client.main()

    print(
        f"STEPS deterministic={determinism.is_enabled()} "
        f"seconds={args.seconds} steps={counter['n']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
