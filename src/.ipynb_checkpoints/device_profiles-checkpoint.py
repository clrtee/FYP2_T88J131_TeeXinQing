from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class DeviceProfile:
    name: str
    power_watt: float
    time_scale: float


PROFILES: Dict[str, DeviceProfile] = {
    "rpi4": DeviceProfile("Raspberry Pi 4", power_watt=7.0, time_scale=2.5),
    "rpi5": DeviceProfile("Raspberry Pi 5", power_watt=12.0, time_scale=1.8),
    "jetson_nano": DeviceProfile("NVIDIA Jetson Nano", power_watt=10.0, time_scale=1.2),
    "coral": DeviceProfile("Google Coral", power_watt=4.0, time_scale=1.4),
    "esp32": DeviceProfile("ESP32", power_watt=0.5, time_scale=50.0),
}
