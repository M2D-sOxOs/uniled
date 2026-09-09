"""UniLED NETwork Device Handler."""

from __future__ import annotations

import asyncio
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
    UNILED_COMMAND_SETTLE_DELAY as UNILED_NET_COMMAND_SETTLE_DELAY,
    UNILED_TRANSPORT_NET,
)
from ..device import UniledChannel, UniledDevice  # noqa: TID252
from ..discovery import UniledDiscovery, discovery_model  # noqa: TID252
from .model import UniledNetModel
from .retrys import _socket_retry

_LOGGER = logging.getLogger(__name__)

UNILED_NET_DEVICE_TIMEOUT: Final = 5.0
UNILED_NET_ERROR_BACKOFF_TIME = 0.1

## How long to keep an unconfirmed user write as the channel state
## (optimistic), before a polled device state is allowed to replace it.
UNILED_NET_WRITE_PENDING_TIMEOUT: Final = 8.0

## How many times an unconfirmed user write is re-sent to the device
## when a successful poll reports a different (stale) device state.
UNILED_NET_WRITE_RESEND_MAX: Final = 2


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
        timeout: float = UNILED_NET_DEVICE_TIMEOUT,
    ) -> None:
        """Init the UniLED Network Device."""
        self._socket: socket.socket | None = None
        self._lock = asyncio.Lock()
        self._timeout: float = timeout
        self._available = False
        self._unavailable_reason = None
        self._model = None
        self._discovery = discovery
        self._pending_writes: dict[
            tuple[int, str], tuple[Any, float, int]
        ] = {}
        self._send_generation: int = 0
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
    def has_pending_writes(self) -> bool:
        """Return whether there are any unconfirmed user writes.

        Expired writes are discarded first, so a device that has
        never acknowledged a user write stops blocking availability.
        """
        self._expire_pending_writes()
        return bool(self._pending_writes)

    def _register_pending_write(
        self, channel: UniledChannel, attr: str, value: Any
    ) -> None:
        """Remember a user written state until the device confirms it."""
        self._pending_writes[(channel.number, attr)] = (
            value,
            time.monotonic(),
            0,
        )

    def _expire_pending_writes(self, now: float | None = None) -> None:
        """Drop pending writes that have exceeded the pending timeout."""
        if not self._pending_writes:
            return
        now = time.monotonic() if now is None else now
        expired = [
            key
            for key, (_, written, _) in self._pending_writes.items()
            if now - written > UNILED_NET_WRITE_PENDING_TIMEOUT
        ]
        for key in expired:
            _LOGGER.debug(
                "%s: Pending write expired: %s",
                self.name,
                self._pending_writes.pop(key),
            )

    def _reconcile_pending_writes(self) -> None:
        """Re-apply pending user writes over a freshly polled state.

        Called after a poll has decoded (and replaced) the channel
        statuses, before any entity callbacks are fired. This keeps
        the user chosen state visible in the UI while the device is
        still catching up (or the write is being retried).
        """
        self._expire_pending_writes()
        for key, (value, written, resends) in list(self._pending_writes.items()):
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
            _LOGGER.debug(
                "%s: Pending write enforced: %s = %s (device has: %s)",
                self.name,
                attr,
                value,
                channel.get(attr, None),
            )
            channel.set(attr, value)

    async def _resend_pending_writes(self) -> None:
        """Re-send pending writes a limited number of times."""
        for key, (value, _, resends) in list(self._pending_writes.items()):
            if resends >= UNILED_NET_WRITE_RESEND_MAX:
                continue
            channel_number, attr = key
            channel = self.channel(channel_number)
            if channel is None:
                self._pending_writes.pop(key, None)
                continue
            _, written, _ = self._pending_writes[key]
            self._pending_writes[key] = (value, written, resends + 1)
            command = self._model.build_command(self, channel, attr, value)
            if not command:
                continue
            _LOGGER.debug(
                "%s: Resending pending write: %s = %s (attempt %s)",
                self.name,
                attr,
                value,
                resends + 2,
            )
            await self.send(command)

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
        self._register_pending_write(channel, attr, state)
        channel.set(attr, state, True)

        await self.send(command, supersede=True)
        return True

    async def async_set_multi_state(self, channel: UniledChannel, **kwargs) -> bool:
        """Set a channel multi attribute states (optimistically)."""
        self._send_generation += 1
        commands = self._model.build_multi_commands(self, channel, **kwargs)
        for attr, state in kwargs.items():
            self._register_pending_write(channel, attr, state)
            channel.set(attr, state)
        if not commands:
            channel.refresh()
            return True
        success = await self.send(commands, supersede=True)
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
        await self._resend_pending_writes()
        return True

    async def stop(self) -> None:
        """Stop the device."""
        if self.available:
            _LOGGER.debug("%s: Stop", self.name)
            self._send_generation += 1
            async with self._lock:
                self._close()
                self.set_unavailable("Stopped")

    async def send(
        self,
        commands: list[bytes] | bytes,
        retry: int | None = None,
        supersede: bool = False,
    ) -> bool:
        """Send command(s) to a device.

        A superseding send invalidates any older send attempt, which
        will then return (unsuccessfully but harmlessly) at its next
        checkpoint. Non-superseding sends (polls, resends) can
        themselves be superseded by newer user writes.
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
                    return await self._execute_commands(commands, generation)
                except Exception as ex:  # noqa: BLE001
                    if generation != self._send_generation:
                        _LOGGER.debug(
                            "%s: Send cancelled, superseded by a newer request",
                            self.name,
                        )
                        return True
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

        raise RuntimeError("Unreachable")

    async def _execute_commands(
        self, commands: list[bytes], generation: int | None = None
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
                if not await self._execute_transaction(command, generation):
                    return False
            await asyncio.sleep(UNILED_NET_COMMAND_SETTLE_DELAY)
        if self._model.close_after_send:
            self._close()
        return True

    async def _execute_transaction(
        self, command: bytes, generation: int | None = None
    ) -> bool:
        """Execute a single command."""
        if not self._send_bytes(command):
            _LOGGER.warning("%s: Command send failed!", self.name)
            return False

        if (expected := self._model.length_response_header(self, command)) == 0:
            return True

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
            timeout_left = self._timeout - (time.monotonic() - begin)
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
        self._socket.settimeout(self._timeout)
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
