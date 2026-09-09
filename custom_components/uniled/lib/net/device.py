"""UniLED NETwork Device Handler."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import logging
import select
import socket
import time
from typing import Any, Final

from ..const import (  # noqa: TID252
    ATTR_UL_IP_ADDRESS,
    ATTR_UL_LOCAL_NAME,
    ATTR_UL_MAC_ADDRESS,
    ATTR_UL_MODEL_NAME,
    ATTR_UL_POWER,
    UNILED_TRANSPORT_NET,
)
from ..device import UniledChannel, UniledDevice  # noqa: TID252
from ..discovery import UniledDiscovery, discovery_model  # noqa: TID252
from .model import UniledNetModel
from .retrys import _socket_retry

_LOGGER = logging.getLogger(__name__)

UNILED_NET_ERROR_BACKOFF_TIME = 0.1

## Aggressive timeouts for links with heavy packet loss spikes:
## connections must be established fast, response reads tolerate
## one TCP level retransmission.
UNILED_NET_CONNECT_TIMEOUT: Final = 0.15
UNILED_NET_RESPONSE_TIMEOUT: Final = 0.4

## Duplicate copies sent back to back for every user write command,
## pushing the per pass delivery probability above 80% on a link
## with 70% packet loss.
UNILED_NET_WRITE_BURST: Final = 5

## Delay between distinct commands (burst duplicates have none).
UNILED_NET_SETTLE_DELAY: Final = 0.05

## Background sync task pacing. Passes start fast and slow down
## while there is no sign of transmission trouble, until a poll
## confirms the written states.
UNILED_NET_SYNC_PASS_DELAY: Final = 0.05
UNILED_NET_SYNC_SLOW_DELAY: Final = 1.0

## Reduced sync pacing once consecutive poll failures have
## determined the device unavailable. Retries keep running so the
## target state lands as soon as the link recovers.
UNILED_NET_SYNC_DEGRADED_DELAY: Final = 1.0

## Consecutive poll failures before the device is reported
## unavailable to Home Assistant.
UNILED_NET_POLL_FAILURES_MAX: Final = 5

## How long after a user write poll failures are ignored, while
## the write may still be reaching the device.
UNILED_NET_WRITE_PENDING_TIMEOUT: Final = 8.0

## Minimum delay between command retries sensor announcements.
UNILED_NET_RETRIES_ANNOUNCE_DELAY: Final = 0.5


##
## UniLed NETwork Device Handler
##
class UniledNetDevice(UniledDevice):
    """UniLED NETwork Device Class."""

    ##
    ## Initialize device instance
    ##
    def __init__(
        self,
        discovery: UniledDiscovery | None,
        options: dict[str, Any] | None = None,
    ) -> None:
        """Init the UniLED Network Device."""
        self._socket: socket.socket | None = None
        self._lock = asyncio.Lock()
        self._available = False
        self._unavailable_reason = None
        self._model = None
        self._discovery = discovery
        self._pending_writes: dict[
            tuple[int, str], tuple[Any, float, list[bytearray]]
        ] = {}
        self._send_generation: int = 0
        self._sync_task: asyncio.Task | None = None
        self._sync_wake: asyncio.Event = asyncio.Event()
        self._sync_trouble: bool = False
        self._sync_pass_delay: float = UNILED_NET_SYNC_PASS_DELAY
        self._command_retries: int = 0
        self._retries_announced: tuple[int, float] | None = None
        super().__init__(options)

        assert discovery is not None

        if self.model is not None:
            _LOGGER.debug(
                "%s: Inititalizing: %s (%s)", self.name, self.model_name, self.address
            )
            self._create_channels()

    @property
    def transport(self) -> str:
        """Return the device transport."""
        return UNILED_TRANSPORT_NET

    @property
    def model(self) -> UniledNetModel:
        """Return the device model."""
        if self._model is None and self._discovery:
            self._model = discovery_model(self._discovery)
        return self._model

    @property
    def model_name(self) -> str | None:
        """Return the device model name."""
        if self.model is not None:
            return self.model.model_name
        if self._discovery:
            return self._discovery.get(ATTR_UL_MODEL_NAME, None)
        return None

    @property
    def name(self) -> str:
        """Get the name of the device."""
        if self._discovery:
            name = self._discovery.get(ATTR_UL_LOCAL_NAME, self.model_name)
            if name is not None:
                return name
        return self.short_address(self.address)

    @property
    def host(self) -> str | None:
        """Get the hostname of the device."""
        if self._discovery:
            return self._discovery.get(ATTR_UL_IP_ADDRESS, None)
        return None

    @property
    def port(self) -> int:
        """Return the network port."""
        assert self.model is not None  # nosec
        return self.model.port

    @property
    def address(self) -> str | None:
        """Return the (mac) address of the device."""
        if self._discovery:
            return self._discovery.get(ATTR_UL_MAC_ADDRESS, None)
        return None

    @property
    def available(self) -> bool:
        """Return if the UniLED device available."""
        if self.model and self._available:
            return True
        return False

    @property
    def discovery(self) -> UniledDiscovery | None:
        """Return the discovery data."""
        return self._discovery

    @property
    def max_poll_failures(self) -> int:
        """Return consecutive failed polls before unavailable."""
        return UNILED_NET_POLL_FAILURES_MAX

    @property
    def command_retries(self) -> int:
        """Return the number of retries for the last user command."""
        return self._command_retries

    @property
    def has_pending_writes(self) -> bool:
        """Return whether there are recent unconfirmed user writes.

        Pending writes persist until the device confirms them (or a
        newer user write supersedes them), but only writes newer than
        UNILED_NET_WRITE_PENDING_TIMEOUT shield the device from poll
        failures.
        """
        now = time.monotonic()
        return any(
            now - written <= UNILED_NET_WRITE_PENDING_TIMEOUT
            for _, (_, written, _) in self._pending_writes.items()
        )

    def _register_pending_write(
        self,
        channel: UniledChannel,
        attr: str,
        value: Any,
        commands: list[bytearray],
    ) -> None:
        """Remember a user written state (and its commands) until
        the device confirms it."""
        self._pending_writes[(channel.number, attr)] = (
            value,
            time.monotonic(),
            commands,
        )

    def _reconcile_pending_writes(self) -> None:
        """Re-apply pending user writes over a freshly polled state.

        Called after a poll has decoded (and replaced) the channel
        statuses, before any entity callbacks are fired. Confirms and
        drops writes the device now matches, keeps the user chosen
        state visible otherwise and flags the sync task to speed up.
        """
        for key, (value, _, _) in list(self._pending_writes.items()):
            channel_number, attr = key
            channel = self.channel(channel_number)
            if channel is None:
                self._pending_writes.pop(key, None)
                continue
            if channel.get(attr, None) == value:
                # Device has confirmed the user state.
                _LOGGER.debug(
                    "%s: Pending write confirmed: %s = %s",
                    self.name,
                    attr,
                    value,
                )
                self._pending_writes.pop(key, None)
                continue
            # Device reports a different (stale) state, keep the user
            # state visible and re-send the write at a fast pace.
            self._sync_trouble = True
            _LOGGER.debug(
                "%s: Pending write enforced: %s = %s (device has: %s)",
                self.name,
                attr,
                value,
                channel.get(attr, None),
            )
            channel.set(attr, value)

    def _ensure_sync_task(self) -> None:
        """Ensure the background write sync task is running."""
        if self._sync_task is None or self._sync_task.done():
            self._sync_wake.set()
            self._sync_task = asyncio.create_task(self._async_sync_targets())

    async def _async_sync_targets(self) -> None:
        """Continuously re-send unconfirmed user writes to the device.

        Runs in parallel with the poll task: every pass re-sends each
        unconfirmed write as a burst of duplicate commands until a
        poll confirms the device state (or a newer user write
        supersedes it). Pacing starts fast and slows down while no
        transmission trouble is seen; once consecutive poll failures
        have determined the device unavailable, retries continue at
        a reduced rate so the target lands as soon as the link
        recovers.
        """
        while True:
            try:
                if not self._pending_writes:
                    self._sync_wake.clear()
                    if not self._pending_writes:
                        await self._sync_wake.wait()
                    continue
                trouble = await self._async_sync_pass()
                self._sync_pass_delay = (
                    UNILED_NET_SYNC_PASS_DELAY
                    if trouble
                    else min(
                        self._sync_pass_delay * 2,
                        UNILED_NET_SYNC_SLOW_DELAY,
                    )
                )
                degraded = self._poll_failures >= UNILED_NET_POLL_FAILURES_MAX
                await asyncio.sleep(
                    UNILED_NET_SYNC_DEGRADED_DELAY
                    if degraded
                    else self._sync_pass_delay
                )
            except asyncio.CancelledError:
                raise
            except Exception as ex:  # noqa: BLE001
                _LOGGER.warning(
                    "%s: Write sync task error: %s", self.name, str(ex)
                )
                await asyncio.sleep(UNILED_NET_SYNC_DEGRADED_DELAY)

    async def _async_sync_pass(self) -> bool:
        """One pass: re-send every pending user write as a burst.

        Returns True when transmission trouble was detected (failed
        sends or a stale device state reported by a poll), which
        keeps the pass pacing at its fastest.
        """
        trouble = self._sync_trouble
        self._sync_trouble = False
        sent = 0
        for key, (_, _, commands) in list(self._pending_writes.items()):
            if not commands:
                continue
            if not await self.send(
                commands, retry=0, burst=UNILED_NET_WRITE_BURST
            ):
                trouble = True
            sent += 1
        if sent:
            self._command_retries += 1
            self._announce_command_retries()
        return trouble

    def _announce_command_retries(self, immediate: bool = False) -> None:
        """Announce the retries count to listeners, throttled."""
        now = time.monotonic()
        value = self._command_retries
        if not immediate:
            last = self._retries_announced
            if last is not None and (
                last[0] == value
                or (now - last[1]) < UNILED_NET_RETRIES_ANNOUNCE_DELAY
            ):
                return
        self._retries_announced = (value, now)
        # Fire raw device callbacks only: reconciliation must run
        # against polled states, never optimistic ones.
        super()._fire_callbacks()

    @discovery.setter
    def discovery(self, value: UniledDiscovery) -> None:
        """Set the discovery data."""
        self._discovery = value

    def set_available(self, reason: str) -> None:
        """Set device as available."""
        _LOGGER.debug("%s: Set available: %s", self.name, reason)
        self._unavailable_reason = None
        self._available = True

    def set_unavailable(self, reason: str) -> None:
        """Set device as unavailable."""
        _LOGGER.debug("%s: Set unavailable: %s", self.name, reason)
        self._unavailable_reason = reason
        self._available = False
        self._close()

    def _fire_callbacks(self) -> None:
        """Fire the callbacks, keeping unconfirmed user writes visible."""
        self._reconcile_pending_writes()
        super()._fire_callbacks()

    async def async_set_state(
        self, channel: UniledChannel, attr: str, state: Any
    ) -> bool:
        """Set a channel attribute state (optimistically)."""
        # Cancel any in-flight (or queued) send attempts, so the new
        # state is written to the device as soon as possible.
        self._send_generation += 1

        # Build the command first, as command generation may depend
        # on the current (pre write) channel status.
        command = self._model.build_command(self, channel, attr, state)
        if not command:
            return False

        # Optimistically apply the user state, so the UI keeps showing
        # it while (and regardless of whether) the send is in progress.
        self._register_pending_write(channel, attr, state, command)
        channel.set(attr, state, True)

        # New user command: per command retries counting starts fresh.
        self._command_retries = 0
        self._announce_command_retries(immediate=True)

        self._ensure_sync_task()
        await self.send(command, supersede=True, burst=UNILED_NET_WRITE_BURST)
        return True

    async def async_set_multi_state(self, channel: UniledChannel, **kwargs) -> bool:
        """Set a channel multi attribute states (optimistically)."""
        self._send_generation += 1
        commands = []
        for attr, state in kwargs.items():
            attr_commands = self._model.build_command(self, channel, attr, state)
            self._register_pending_write(channel, attr, state, attr_commands)
            channel.set(attr, state)
            commands.extend(attr_commands)
        # New user command: per command retries counting starts fresh.
        self._command_retries = 0
        self._announce_command_retries(immediate=True)
        self._ensure_sync_task()
        if not commands:
            channel.refresh()
            return True
        success = await self.send(
            commands, supersede=True, burst=UNILED_NET_WRITE_BURST
        )
        channel.refresh()
        return success

    async def startup(self, event=None) -> bool:
        """Startup the device."""
        if not self._started:
            try:
                success = await self.update(retry=3)
                _LOGGER.debug("%s: Startup state: %s", self.name, success)
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning(
                    "%s: Startup - Failed, exception: %s", self.name, str(exc)
                )
                success = False
            self._started = success
        return self._started

    async def update(self, retry: int | None = None) -> bool:
        """Update the device."""
        _LOGGER.debug("%s: Update!", self.name)
        if not (query := self.model.build_state_query(self)):
            raise Exception("Update - Failed: no state query command available!")  # noqa: TRY002
        if not await self.send(query, retry):
            return False
        valid = 0
        for channel in self.channel_list:
            if channel.status.has(ATTR_UL_POWER):
                valid += 1
            _LOGGER.debug(
                "%s: %s - Status: %s",
                self.name,
                channel.identity,
                channel.status.dump(),
            )

        if valid != self.channels:
            _LOGGER.warning("%s: Invalid channel status", self.name)
            return False
        self._ensure_sync_task()
        return True

    async def stop(self) -> None:
        """Stop the device."""
        self._send_generation += 1
        if (task := self._sync_task) is not None:
            self._sync_task = None
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        self._pending_writes.clear()
        if self.available:
            _LOGGER.debug("%s: Stop", self.name)
            async with self._lock:
                self._close()
                self.set_unavailable("Stopped")

    async def send(
        self,
        commands: list[bytes] | bytes,
        retry: int | None = None,
        supersede: bool = False,
        burst: int = 1,
    ) -> bool:
        """Send command(s) to a device.

        Fire and forget commands (no response expected) are sent as a
        burst of duplicate copies for lossy links. A superseding send
        invalidates any older send attempt, which will then return
        (successfully but harmlessly) at its next checkpoint. Non
        superseding sends (polls, sync retries) can themselves be
        superseded by newer user writes.
        """

        if not commands:
            _LOGGER.debug("%s: Send command ignored, no data to send", self.name)
            return False

        if not isinstance(commands, list):
            commands = [commands]

        if retry is None:
            retry = self.retry_count

        max_attempts = retry + 1

        if supersede:
            # Invalidate any send attempt currently holding (or waiting
            # for) the lock, this one takes over as soon as they yield.
            self._send_generation += 1
        generation = self._send_generation

        if self._lock.locked():
            _LOGGER.debug(
                "%s: Operation already in progress, waiting for it to complete",
                self.name,
            )

        async with self._lock:
            for attempt in range(max_attempts):
                if generation != self._send_generation:
                    _LOGGER.debug(
                        "%s: Send cancelled, superseded by a newer request",
                        self.name,
                    )
                    return True
                try:
                    success = await self._execute_commands(
                        commands, generation, burst
                    )
                except Exception as ex:  # noqa: BLE001
                    if generation != self._send_generation:
                        _LOGGER.debug(
                            "%s: Send cancelled, superseded by a newer request",
                            self.name,
                        )
                        return True
                    # Transmission trouble: the sync task speeds up.
                    self._sync_trouble = True
                    if attempt == retry:
                        _LOGGER.error(
                            "%s: Communication failed: %s, stopping trying!",
                            self.name,
                            str(ex),
                        )
                        return False
                    _LOGGER.debug(
                        "%s: Communication failed with: %s, retry attempt %s of %s",
                        self.name,
                        str(ex),
                        attempt + 1,
                        max_attempts,
                    )
                    self._close()
                    await asyncio.sleep(UNILED_NET_ERROR_BACKOFF_TIME)
                    continue
                if success:
                    # Command traffic proves the device is reachable,
                    # a user command counts as one successful poll.
                    self._poll_failures = 0
                else:
                    self._sync_trouble = True
                return success

        raise RuntimeError("Unreachable")

    async def _execute_commands(
        self,
        commands: list[bytes],
        generation: int | None = None,
        burst: int = 1,
    ) -> bool:
        """Execute command(s)."""
        self._connect_if_disconnected()
        for command in commands:
            if generation is not None and generation != self._send_generation:
                _LOGGER.debug(
                    "%s: Command cancelled, superseded by a newer request",
                    self.name,
                )
                return True
            if self.available and command:
                if not await self._execute_transaction(command, generation, burst):
                    return False
                await asyncio.sleep(UNILED_NET_SETTLE_DELAY)
        if self._model.close_after_send:
            self._close()
        return True

    async def _execute_transaction(
        self, command: bytes, generation: int | None = None, burst: int = 1
    ) -> bool:
        """Execute a single command."""
        if (expected := self._model.length_response_header(self, command)) == 0:
            # Fire and forget: send a burst of duplicate copies so a
            # single one getting through applies the state.
            for _ in range(max(1, burst)):
                if generation is not None and generation != self._send_generation:
                    _LOGGER.debug(
                        "%s: Command cancelled, superseded by a newer request",
                        self.name,
                    )
                    return True
                if not self._send_bytes(command):
                    _LOGGER.warning("%s: Command send failed!", self.name)
                    return False
            return True
        if not self._send_bytes(command):
            _LOGGER.warning("%s: Command send failed!", self.name)
            return False

        header = await self._async_read_bytes(expected, generation)
        if len(header) != expected:
            _LOGGER.warning(
                "%s: Response Header Error: read %d, expected %d",
                self.name,
                len(header),
                expected,
            )
            return False
        expected = self._model.decode_response_header(self, command, header)
        if expected == -1:
            _LOGGER.warning("%s: Response Header Error!", self.name)
            return False
        if expected is None or expected == 0:
            return True

        payload = await self._async_read_bytes(expected, generation)
        if len(payload) != expected:
            _LOGGER.warning(
                "%s: Response Payload Error: read %d, expected %d",
                self.name,
                len(payload),
                expected,
            )
            return False
        try:
            if (
                self._model.decode_response_payload(self, command, header, payload)
                is True
            ):
                _LOGGER.debug("%s: Transaction successful", self.name)
                self._fire_callbacks()
                return True
            _LOGGER.debug("%s: Transaction failed", self.name)
        except Exception:  # noqa: BLE001
            _LOGGER.warning(
                "%s: Response decoder exception!",
                self.name,
                exc_info=True,
            )
        return False

    @_socket_retry(attempts=2)
    def _send_bytes(self, bytes: bytearray) -> bool:
        assert self._socket is not None
        _LOGGER.debug(
            "%s => %s (%d)",
            self.name,
            "".join(f"{x:02X}" for x in bytes),
            len(bytes),
        )
        if self._socket.sendall(bytes) is None:
            return True
        return False

    async def _async_read_bytes(
        self, expected: int, generation: int | None = None
    ) -> bytearray:
        assert self._socket is not None
        remaining = expected
        rx = bytearray()
        begin = time.monotonic()
        while remaining > 0:
            if generation is not None and generation != self._send_generation:
                _LOGGER.debug(
                    "%s: Read cancelled, superseded by a newer request", self.name
                )
                break
            timeout_left = UNILED_NET_RESPONSE_TIMEOUT - (time.monotonic() - begin)
            if timeout_left <= 0:
                break
            try:
                self._socket.setblocking(False)
                read_ready, _, _ = select.select([self._socket], [], [], timeout_left)
                if not read_ready:
                    _LOGGER.debug("%s: timed out reading %d bytes", self.name, expected)
                    break
                chunk = self._socket.recv(remaining)
                chunk_size = len(chunk) if chunk else 0
                if chunk:
                    _LOGGER.debug(
                        "%s <= %s (%d)",
                        self.name,
                        "".join(f"{x:02X}" for x in chunk),
                        chunk_size,
                    )
                    begin = time.monotonic()
                elif chunk_size == 0:
                    await asyncio.sleep(UNILED_NET_ERROR_BACKOFF_TIME)
                remaining -= chunk_size
                rx.extend(chunk)
            except OSError as ex:
                _LOGGER.debug("%s: socket error (%s): %s", self.name, self.host, ex)
            finally:
                self._socket.setblocking(True)
        return rx

    def _connect_if_disconnected(self) -> None:
        """Connect only if not already connected."""
        if self._socket is None:
            self._connect()

    @_socket_retry(attempts=0)
    def _connect(self) -> None:
        self._close()
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(UNILED_NET_CONNECT_TIMEOUT)
        self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        _LOGGER.debug("%s: Connect %s:%d", self.name, self.host, self.port)
        self._socket.connect((self.host, self.port))

    def _close(self) -> None:
        if self._socket is None:
            return
        try:
            self._socket.close()
        except OSError:
            pass
        finally:
            self._socket = None
        _LOGGER.debug("%s: Socket closed", self.name)
