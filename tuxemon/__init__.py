# SPDX-License-Identifier: GPL-3.0
# Copyright (c) 2014-2026 William Edwards <shadowapex@gmail.com>, Benjamin Bean <superman2k5@gmail.com>
from tuxemon.version import __version__, version_info

# Opt-in determinism shim.  This is a no-op unless TUXEMON_DETERMINISTIC is
# set, and it must run before any module caches a reference to time.time or
# time.perf_counter, hence its placement in the package __init__.
from tuxemon import determinism as _determinism

_determinism.install()

__all__ = ["__version__", "version_info"]
