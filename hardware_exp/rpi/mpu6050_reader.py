"""
mpu6050_reader.py  —  run on RASPBERRY PI
MPU6050 IMU driver over I2C (smbus2).
Outputs 6-channel data matching UCI-HAR inertial signal format:
  [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]

Requires:  pip install smbus2
Hardware:  MPU6050 wired to RPi I2C bus 1 (pins 3=SDA, 5=SCL)
           I2C must be enabled: raspi-config → Interface → I2C → Yes
"""

import time
import numpy as np

try:
    import smbus2
    _HW_AVAILABLE = True
except ImportError:
    _HW_AVAILABLE = False
    print("[mpu6050] smbus2 not found — using simulated data")


class MPU6050:
    # I2C address (AD0 pin → GND = 0x68, AD0 → 3.3V = 0x69)
    ADDR = 0x68

    # Register map
    _PWR_MGMT_1   = 0x6B
    _SMPLRT_DIV   = 0x19
    _CONFIG       = 0x1A
    _GYRO_CONFIG  = 0x1B
    _ACCEL_CONFIG = 0x1C
    _ACCEL_XOUT_H = 0x3B
    _GYRO_XOUT_H  = 0x43

    # Scale factors for default ±2g / ±250°/s ranges
    _ACCEL_LSB = 16384.0   # counts per g
    _GYRO_LSB  = 131.0     # counts per °/s

    def __init__(self, bus: int = 1, addr: int = 0x68):
        self.addr = addr
        self._sim = not _HW_AVAILABLE

        if not self._sim:
            self._bus = smbus2.SMBus(bus)
            self._init_sensor()
        else:
            print("[mpu6050] Simulation mode active")

    def _write(self, reg: int, val: int):
        self._bus.write_byte_data(self.addr, reg, val)

    def _read_word(self, reg: int) -> int:
        high = self._bus.read_byte_data(self.addr, reg)
        low  = self._bus.read_byte_data(self.addr, reg + 1)
        val  = (high << 8) | low
        return val - 65536 if val > 32767 else val

    def _init_sensor(self):
        self._write(self._PWR_MGMT_1, 0x00)     # Wake up, use internal oscillator
        time.sleep(0.05)
        # 50 Hz sample rate: SMPLRT_DIV = (1000 / 50) - 1 = 19
        self._write(self._SMPLRT_DIV,   0x13)
        # DLPF bandwidth ~21 Hz (config=4) — reduces high-freq noise
        self._write(self._CONFIG,       0x04)
        # ±2g accelerometer range (default)
        self._write(self._ACCEL_CONFIG, 0x00)
        # ±250 °/s gyroscope range (default)
        self._write(self._GYRO_CONFIG,  0x00)
        time.sleep(0.1)
        print(f"[mpu6050] Initialised at I2C 0x{self.addr:02X}, 50 Hz, ±2g/±250°/s")

    def read(self) -> np.ndarray:
        """Read one sample: [ax, ay, az, gx, gy, gz]."""
        if self._sim:
            # Gaussian noise centred on standing-still values
            acc  = np.random.randn(3) * 0.02 + [0.0, 0.0, 1.0]
            gyro = np.random.randn(3) * 0.05
            return np.concatenate([acc, gyro]).astype(np.float32)

        ax = self._read_word(self._ACCEL_XOUT_H)     / self._ACCEL_LSB
        ay = self._read_word(self._ACCEL_XOUT_H + 2) / self._ACCEL_LSB
        az = self._read_word(self._ACCEL_XOUT_H + 4) / self._ACCEL_LSB
        gx = self._read_word(self._GYRO_XOUT_H)      / self._GYRO_LSB
        gy = self._read_word(self._GYRO_XOUT_H + 2)  / self._GYRO_LSB
        gz = self._read_word(self._GYRO_XOUT_H + 4)  / self._GYRO_LSB
        return np.array([ax, ay, az, gx, gy, gz], dtype=np.float32)

    def collect_window(self, window_size: int = 128, fs: int = 50) -> np.ndarray:
        """
        Collect one fixed-length window at target sampling rate.
        Returns shape (window_size, 6) — transpose to (6, window_size) for 1D-CNN.
        Timing accuracy: ~±2 ms per sample on RPi 4B.
        """
        samples = []
        interval = 1.0 / fs
        for _ in range(window_size):
            t0 = time.perf_counter()
            samples.append(self.read())
            elapsed = time.perf_counter() - t0
            remaining = interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
        return np.stack(samples)   # (128, 6)


if __name__ == "__main__":
    mpu = MPU6050()
    print("Reading 5 samples at 50 Hz...")
    for i in range(5):
        sample = mpu.read()
        print(f"  [{i}] acc={sample[:3].round(3)}  gyro={sample[3:].round(3)}")
        time.sleep(0.02)
    print("\nCollecting one 128-sample window (~2.56 s)...")
    window = mpu.collect_window()
    print(f"Window shape: {window.shape}  (should be (128, 6))")
    print(f"Acc mean: {window[:, :3].mean(0).round(4)}")
    print(f"Gyro mean:{window[:, 3:].mean(0).round(4)}")
