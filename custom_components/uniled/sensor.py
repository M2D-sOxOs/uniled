"""Platform for UniLED number integration."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    # SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import SIGNAL_STRENGTH_DECIBELS_MILLIWATT
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import ATTR_UL_MAC_ADDRESS as EXTRA_ATTRIBUTE_MAC_ADDRESS
from .entity import (
    AddEntitiesCallback,
    Platform,
    UniledChannel,
    UniledEntity,
    UniledUpdateCoordinator,
    async_uniled_entity_setup,
)
from .lib.attributes import SensorAttribute, UniledAttribute, UniledGroup
from .lib.net.device import UNILED_TRANSPORT_NET

_LOGGER = logging.getLogger(__name__)

PARALLEL_UPDATES = 1


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the UniLED sensor platform."""
    await async_uniled_entity_setup(
        hass, entry, async_add_entities, _add_sensor_entity, Platform.SENSOR
    )


def _add_sensor_entity(
    coordinator: UniledUpdateCoordinator,
    channel: UniledChannel,
    feature: UniledAttribute | None,
) -> UniledEntity | list[UniledEntity] | None:
    """Create UniLED sensor entity."""
    if feature:
        return UniledSensorEntity(coordinator, channel, feature)
    if channel.number == 0:
        if coordinator.device.transport == UNILED_TRANSPORT_NET:
            return [UniledCommandRetriesSensor(coordinator, channel)]
        return UniledSignalSensor(coordinator, channel)
    return None


class UniledSensorEntity(
    UniledEntity, CoordinatorEntity[UniledUpdateCoordinator], SensorEntity
):
    """Defines a UniLED sensor."""

    def __init__(
        self,
        coordinator: UniledUpdateCoordinator,
        channel: UniledChannel,
        feature: SensorAttribute,
    ) -> None:
        """Initialize a UniLED sensor."""
        super().__init__(coordinator, channel, feature)

    @property
    def native_value(self) -> str:
        """Return the value reported by the sensor."""
        value = self.device.get_state(self.channel, self.feature.attr)
        if isinstance(value, str):
            value = value.lower()
        return value


@dataclass
class RSSIFeature(SensorAttribute):
    """UniLED RSSI Feature Class."""

    def __init__(self) -> None:
        """Initialize RSSI Feature."""
        super().__init__(
            None,
            "RSSI",
            "mdi:signal",
            key="rssi",
            group=UniledGroup.DIAGNOSTIC,
            enabled=False,
        )


@dataclass
class CommandRetriesFeature(SensorAttribute):
    """UniLED Command Retries Feature Class."""

    def __init__(self) -> None:
        """Initialize Command Retries Feature."""
        super().__init__(
            None,
            "Command Retries",
            "mdi:repeat",
            key="command_retries",
            group=UniledGroup.DIAGNOSTIC,
            enabled=True,
        )


class UniledCommandRetriesSensor(
    UniledEntity, CoordinatorEntity[UniledUpdateCoordinator], SensorEntity
):
    """Defines a UniLED command retries sensor."""

    def __init__(
        self,
        coordinator: UniledUpdateCoordinator,
        channel: UniledChannel,
    ) -> None:
        """Initialize the command retries sensor."""
        super().__init__(coordinator, channel, CommandRetriesFeature())

    @callback
    def _async_update_attrs(self, first: bool = False) -> None:
        """Handle updating _attr values."""
        super()._async_update_attrs()
        self._attr_state_class = SensorStateClass.MEASUREMENT

    @property
    def native_value(self) -> int:
        """Return the value reported by the sensor."""
        return self.device.command_retries

    async def async_added_to_hass(self) -> None:
        """Handle entity which will be added."""
        await super().async_added_to_hass()
        # Sync retries are announced by the device write sync task,
        # between polls, via device level callbacks.
        self.async_on_remove(
            self.device.register_callback(self._handle_coordinator_update)
        )


class UniledSignalSensor(
    UniledEntity, CoordinatorEntity[UniledUpdateCoordinator], SensorEntity
):
    """Defines a UniLED Signal Sensor control."""

    _unrecorded_attributes = frozenset(
        {
            EXTRA_ATTRIBUTE_MAC_ADDRESS,
        }
    )

    def __init__(
        self,
        coordinator: UniledUpdateCoordinator,
        channel: UniledChannel,
    ) -> None:
        """Initialize a UniLED effect speed control."""
        super().__init__(coordinator, channel, RSSIFeature())

    @callback
    def _async_update_attrs(self, first: bool = False) -> None:
        """Handle updating _attr values."""
        super()._async_update_attrs()
        self._attr_device_class = SensorDeviceClass.SIGNAL_STRENGTH
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_native_unit_of_measurement = SIGNAL_STRENGTH_DECIBELS_MILLIWATT

    @property
    def native_value(self) -> str:
        """Return the value reported by the sensor."""
        return self.device.rssi

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the optional state attributes."""
        return {EXTRA_ATTRIBUTE_MAC_ADDRESS: self.device.address}
