# SPDX-License-Identifier: GPL-3.0
"""
Opt-in determinism shim.

Enabled only when the environment variable ``TUXEMON_DETERMINISTIC`` is set to
something other than ``""``/``0``/``false``.  When disabled every function here
is a no-op and the normal game is byte-for-byte unaffected.

When enabled:

* a *virtual* clock replaces the wall clock (``time.time``, ``time.perf_counter``,
  ``time.monotonic`` and their ``_ns`` variants).  The virtual clock only moves
  when :func:`advance` is called, i.e. once per simulation step;
* ``time.sleep`` becomes a no-op;
* the global :mod:`random` stream is seeded from ``TUXEMON_SEED``;
* ``uuid.uuid4`` is replaced by a separately seeded generator so entity ids do
  not consume (or depend on) OS entropy;
* :meth:`tuxemon.time_handler.TimeHandler.get_current_time` is pinned to the
  virtual clock so the in-game day/night cycle and season are reproducible.

The main loop gate lives in ``tuxemon/client.py``.
"""
from __future__ import annotations

import os
import random as _random
import time as _time
import uuid as _uuid
from datetime import datetime, timezone

__all__ = (
    "is_enabled",
    "install",
    "now",
    "advance",
    "set_time",
    "seed",
)

_FALSEY = ("", "0", "false", "no", "off")


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in _FALSEY


_ENABLED: bool = _env_flag("TUXEMON_DETERMINISTIC")
_SEED: int = int(os.environ.get("TUXEMON_SEED", "20260101"))

# Diagnostic knobs: let a bisect turn individual pieces of the shim off so it
# is possible to attribute a divergence to a specific source.
_PATCH_CLOCK: bool = os.environ.get("TUXEMON_DET_CLOCK", "1") not in _FALSEY
_PATCH_RANDOM: bool = os.environ.get("TUXEMON_DET_RANDOM", "1") not in _FALSEY
_PATCH_UUID: bool = os.environ.get("TUXEMON_DET_UUID", "1") not in _FALSEY

# 2026-06-15 12:00:00 UTC -- a fixed, boring "daytime" wall clock origin.
_EPOCH: float = 1781524800.0

_virtual: float = 0.0
_installed: bool = False

# real functions, kept so install() is idempotent and reversible
_real = {
    "time": _time.time,
    "perf_counter": _time.perf_counter,
    "monotonic": _time.monotonic,
    "time_ns": _time.time_ns,
    "perf_counter_ns": _time.perf_counter_ns,
    "monotonic_ns": _time.monotonic_ns,
    "sleep": _time.sleep,
    "uuid4": _uuid.uuid4,
}


def is_enabled() -> bool:
    """True when the determinism shim is active."""
    return _ENABLED


def seed() -> int:
    return _SEED


def now() -> float:
    """Current virtual timestamp (unix seconds)."""
    return _EPOCH + _virtual


def advance(dt: float) -> None:
    """Move the virtual clock forward by ``dt`` seconds. No-op when disabled."""
    global _virtual
    if _ENABLED:
        _virtual += dt


def set_time(value: float) -> None:
    global _virtual
    _virtual = value


def _v_time() -> float:
    return _EPOCH + _virtual


def _v_perf() -> float:
    return _virtual


def _v_time_ns() -> int:
    return int((_EPOCH + _virtual) * 1e9)


def _v_perf_ns() -> int:
    return int(_virtual * 1e9)


def _v_sleep(_seconds: float) -> None:
    return None


def _make_uuid4_source() -> "_random.Random":
    return _random.Random(_SEED ^ 0x5F5F5F5F)


def install() -> bool:
    """
    Install the shim. Safe to call multiple times and from anywhere; returns
    True if the shim is (now) active.
    """
    global _installed
    if not _ENABLED or _installed:
        return _ENABLED

    if _PATCH_CLOCK:
        _time.time = _v_time  # type: ignore[assignment]
        _time.perf_counter = _v_perf  # type: ignore[assignment]
        _time.monotonic = _v_perf  # type: ignore[assignment]
        _time.time_ns = _v_time_ns  # type: ignore[assignment]
        _time.perf_counter_ns = _v_perf_ns  # type: ignore[assignment]
        _time.monotonic_ns = _v_perf_ns  # type: ignore[assignment]
        _time.sleep = _v_sleep  # type: ignore[assignment]
        _pin_time_handler()

    if _PATCH_UUID:
        _uuid_rng = _make_uuid4_source()

        def _v_uuid4() -> _uuid.UUID:
            return _uuid.UUID(int=_uuid_rng.getrandbits(128), version=4)

        _uuid.uuid4 = _v_uuid4  # type: ignore[assignment]

    if _PATCH_RANDOM:
        _random.seed(_SEED)

    _installed = True
    return True


def _pin_time_handler() -> None:
    """Pin the in-game real-world clock to the virtual clock."""
    try:
        from tuxemon.time_handler import TimeHandler
    except Exception:  # pragma: no cover - import cycles / partial installs
        return

    def get_current_time(self: TimeHandler) -> datetime:
        return datetime.fromtimestamp(_v_time(), tz=timezone.utc).replace(
            tzinfo=None
        )

    TimeHandler.get_current_time = get_current_time  # type: ignore[method-assign]


def uninstall() -> None:
    """Restore the real clock (used by tests)."""
    global _installed
    if not _installed:
        return
    _time.time = _real["time"]  # type: ignore[assignment]
    _time.perf_counter = _real["perf_counter"]  # type: ignore[assignment]
    _time.monotonic = _real["monotonic"]  # type: ignore[assignment]
    _time.time_ns = _real["time_ns"]  # type: ignore[assignment]
    _time.perf_counter_ns = _real["perf_counter_ns"]  # type: ignore[assignment]
    _time.monotonic_ns = _real["monotonic_ns"]  # type: ignore[assignment]
    _time.sleep = _real["sleep"]  # type: ignore[assignment]
    _uuid.uuid4 = _real["uuid4"]  # type: ignore[assignment]
    _installed = False
