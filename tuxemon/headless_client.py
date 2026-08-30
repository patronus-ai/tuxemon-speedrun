# SPDX-License-Identifier: GPL-3.0
# Copyright (c) 2014-2026 William Edwards <shadowapex@gmail.com>, Benjamin Bean <superman2k5@gmail.com>
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from tuxemon import determinism
from tuxemon.base_client import BaseClient, ClientState

if TYPE_CHECKING:
    from tuxemon.config import TuxemonConfig
    from tuxemon.prepare import DisplayContext

logger = logging.getLogger(__name__)


class HeadlessClient(BaseClient):
    """
    Headless client for server-side processing of game logic.
    This client runs without graphics, only handling events and
    game state updates.

    Parameters:
        config: The configuration for the game.
    """

    def __init__(self, config: TuxemonConfig, context: DisplayContext) -> None:
        super().__init__(config, context)
        self.step_count = 0

    def tick(self, dt: float = 1.0 / 60.0) -> None:
        """Advance the simulation by exactly one fixed step."""
        self.update(dt)
        determinism.advance(dt)
        self.step_count += 1

    def main(self) -> None:
        FIXED_DT = 1.0 / 60.0

        if determinism.is_enabled():
            # one step per iteration; the wall clock never decides step count
            while self.state != ClientState.DONE:
                if self.state == ClientState.RUNNING:
                    self.tick(FIXED_DT)
                elif self.state == ClientState.EXITING:
                    self.perform_cleanup()
                    self.state = ClientState.DONE
            return

        accumulator = 0.0
        last_time = time.time()

        while self.state != ClientState.DONE:
            if self.state == ClientState.RUNNING:
                now = time.time()
                frame_time = now - last_time
                last_time = now

                if frame_time > 0.25:
                    frame_time = 0.25

                accumulator += frame_time

                while accumulator >= FIXED_DT:
                    self.update(FIXED_DT)
                    accumulator -= FIXED_DT

                time.sleep(0.001)

            elif self.state == ClientState.EXITING:
                self.perform_cleanup()
                self.state = ClientState.DONE

    def queue_command(self, command: Callable[[], None]) -> None:
        self.command_queue.put(command)
        logger.debug("Queued command for execution in main thread.")

    def update(self, dt: float) -> None:
        """Main loop for entire game."""
        self.update_states(dt)
