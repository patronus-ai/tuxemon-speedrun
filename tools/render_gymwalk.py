#!/usr/bin/env python3
"""Render the gymwalk baseline to video.

WHY A WRAPPER, NOT A REIMPLEMENTATION. det_probe_spyder.main() holds the whole driver loop and
exposes no callback. Copying that loop here would mean the video is of a re-run route rather than
the measured one -- and this route is 6 attempts, 5 defeat-warps and 23 battles deep, so "close
enough" would not be. Instead: monkeypatch LocalPygameClient.tick to capture the surface after
each step, then call the probe's own main(). Same code path, same digest.

Drawing is OFF in the probe (rendering costs time the digest does not need), so
TUXEMON_DETERMINISTIC_DRAW=1 is forced here.

EVERY Nth STEP. 105,397 steps is ~29 min at 1x; --every 4 gives ~26k frames and a ~7 min
4x-speed video, piped straight to ffmpeg with no PNGs on disk.
"""
import argparse, os, sys, subprocess
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ["TUXEMON_DETERMINISTIC_DRAW"] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--every", type=int, default=4)
ap.add_argument("--fps", type=int, default=60)
ap.add_argument("--scenario", default="gymwalk")
ap.add_argument("--steps", type=int, default=None)
a, rest = ap.parse_known_args()

import pygame
from tuxemon.client import LocalPygameClient
import det_probe_spyder as P

state = {"n": 0, "ff": None}
_orig_update = LocalPygameClient.update

def update(self, dt):
    # The probe drives client.update(FIXED_DT) directly (det_probe_spyder.py:816), not tick(),
    # so patching tick captured nothing. Draw after each update so the frame reflects the step.
    _orig_update(self, dt)
    # client.draw() is what _deterministic_main calls when TUXEMON_DETERMINISTIC_DRAW=1.
    # renderer.draw() and state_drawer.draw() both skip the map layer -- 94.5% black.
    self.draw()
    n = state["n"]; state["n"] = n + 1
    if n % a.every:
        return
    surf = getattr(self, "screen", None)
    if surf is None:
        return
    if state["ff"] is None:
        w, h = surf.get_size()
        state["ff"] = subprocess.Popen(
            ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
             "-framerate", str(a.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "20", a.out],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  capturing {w}x{h}, every {a.every} steps", flush=True)
    try:
        state["ff"].stdin.write(pygame.image.tostring(surf, "RGB"))
    except BrokenPipeError:
        pass

LocalPygameClient.update = update

argv = ["det_probe_spyder", "--scenario", a.scenario, "--out", "/tmp/render_probe_out.json"]
if a.steps: argv += ["--steps", str(a.steps)]
sys.argv = argv + rest
try:
    P.main()
finally:
    if state["ff"]:
        state["ff"].stdin.close(); state["ff"].wait()
    print(f"  captured {state['n']//a.every} frames -> {a.out}", flush=True)
