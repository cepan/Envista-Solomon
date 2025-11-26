import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

try:
    import serial  # type: ignore
    from serial.tools import list_ports  # type: ignore
except Exception:
    serial = None
    list_ports = None


@dataclass
class LinearAxisResult:
    success: bool
    message: str
    position_mm: Optional[float] = None


class _LinearAxis:
    """
    Serial controller for the FUYU linear axis driven by the Arduino sketch
    provided in the project description.

    Protocol (9600 8N1):
      - 'C'          -> calibrate (left then right); prints [CAL] ... lines.
      - 'G <mm>'     -> go to absolute position 0..100 mm; prints [MOVE] ... lines.
      - 'R'          -> print calibration info (optional).
      - 'S'          -> stop motion (optional).
    """

    def __init__(self):
        self._ser: Optional[serial.Serial] = None if serial else None
        self._lock = threading.Lock()
        self._guard_ms = 20
        self._last_cmd_ts = 0.0
        self._reader: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._listeners: List[Callable[[str], None]] = []
        self.calibrated: bool = False
        self.position_mm: Optional[float] = None

    # Discovery
    def list_ports(self) -> List[str]:
        if list_ports is None:
            return []
        return [p.device for p in list_ports.comports()]

    # Connection
    def connect(self, port: str, baud: int = 9600) -> bool:
        if serial is None:
            raise RuntimeError("pyserial not installed. Install with 'pip install pyserial'.")
        self.disconnect()
        try:
            ser = serial.Serial(
                port=port,
                baudrate=baud,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1,
                write_timeout=0.2,
            )
            self._ser = ser
            self.calibrated = False
            self.position_mm = None
            self._start_reader()
            self._emit(f"[AXIS] Connected to {port}")
            return True
        except Exception:
            self.disconnect()
            return False

    def disconnect(self) -> None:
        self._stop_reader()
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
        self._ser = None
        self.calibrated = False
        self.position_mm = None

    def is_connected(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def port_name(self) -> Optional[str]:
        return self._ser.port if self._ser else None

    # Messaging
    def add_listener(self, cb: Callable[[str], None]) -> None:
        self._listeners.append(cb)

    def remove_listener(self, cb: Callable[[str], None]) -> None:
        try:
            self._listeners.remove(cb)
        except ValueError:
            pass

    def _emit(self, msg: str) -> None:
        for cb in list(self._listeners):
            try:
                cb(msg)
            except Exception:
                pass

    def _start_reader(self) -> None:
        if self._reader and self._reader.is_alive():
            return
        self._stop.clear()
        self._reader = threading.Thread(target=self._reader_loop, name="axis_reader", daemon=True)
        self._reader.start()

    def _stop_reader(self) -> None:
        self._stop.set()
        if self._reader and self._reader.is_alive():
            self._reader.join(timeout=0.5)
        self._reader = None

    def _reader_loop(self) -> None:
        ser = self._ser
        if ser is None:
            return
        while not self._stop.is_set():
            try:
                line = ser.readline()
                if not line:
                    continue
                text = line.decode("ascii", errors="ignore").strip()
                if text:
                    self._emit(text)
            except Exception:
                # Stop reader on error; caller can reconnect
                break

    def _send(self, cmd: str) -> None:
        if not self.is_connected():
            raise RuntimeError("Linear axis not connected.")
        now = time.time() * 1000.0
        with self._lock:
            delta = now - self._last_cmd_ts
            if delta < self._guard_ms:
                time.sleep((self._guard_ms - delta) / 1000.0)
            payload = (cmd + "\n").encode("ascii")
            self._ser.write(payload)
            self._ser.flush()
            self._last_cmd_ts = time.time() * 1000.0

    # High-level operations
    def calibrate_and_home(self, *, timeout: float = 120.0, home_mm: float = 50.0) -> LinearAxisResult:
        """
        Run the 'C' calibration sequence, then move to the requested home position.
        """
        if not self.is_connected():
            return LinearAxisResult(False, "Linear axis not connected.", self.position_mm)

        self.calibrated = False

        done = threading.Event()
        error: List[Optional[str]] = [None]

        def handler(msg: str) -> None:
            up = msg.upper()
            if up.startswith("[CAL ERR]"):
                error[0] = msg
                done.set()
                return
            if "CALIBRATION COMPLETE" in up:
                done.set()

        self.add_listener(handler)
        try:
            self._emit("[AXIS] Starting calibration...")
            self._send("C")
            if not done.wait(timeout):
                raise TimeoutError("Calibration timed out.")
            if error[0]:
                raise RuntimeError(error[0])
        except Exception as ex:
            self.calibrated = False
            return LinearAxisResult(False, f"[CAL] Failed: {ex}", self.position_mm)
        finally:
            self.remove_listener(handler)

        self.calibrated = True

        # After calibration, Arduino is at right limit (100 mm); go to configured home.
        move_res = self._go_to_internal(home_mm, timeout=max(10.0, timeout / 2.0))
        if move_res.success:
            msg = f"[CAL] Calibration complete; homed to {home_mm:.1f} mm."
        else:
            msg = f"[CAL] Calibration complete, but homing to {home_mm:.1f} mm failed: " + move_res.message
        return LinearAxisResult(move_res.success, msg, move_res.position_mm)

    def read_calibration_info(self, *, timeout: float = 8.0, quiet_time: float = 0.3) -> LinearAxisResult:
        """
        Send 'R' and wait until its output finishes (quiet for quiet_time).
        Uses the response to refresh calibrated flag and last known position.
        """
        if not self.is_connected():
            return LinearAxisResult(False, "Linear axis not connected.", self.position_mm)

        seen = [False]
        last_ts = [time.time()]
        info_pos: List[Optional[float]] = [None]
        info_cal: List[Optional[bool]] = [None]

        def handler(msg: str) -> None:
            seen[0] = True
            last_ts[0] = time.time()
            up = msg.upper()
            try:
                if "CALIBRATED" in up:
                    if "YES" in up:
                        info_cal[0] = True
                    elif "NO" in up:
                        info_cal[0] = False
                if "CURRENTPOS" in up or "CURRENT POSITION" in up:
                    import re

                    m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*MM", up)
                    if m:
                        info_pos[0] = float(m.group(1))
            except Exception:
                pass

        self.add_listener(handler)
        try:
            self._send("R")
            start = time.time()
            while time.time() - start < timeout:
                if seen[0] and (time.time() - last_ts[0]) >= quiet_time:
                    break
                time.sleep(0.05)
            if not seen[0]:
                raise TimeoutError("No response to 'R' command.")
        except Exception as ex:
            return LinearAxisResult(False, f"[INFO] Read failed: {ex}", self.position_mm)
        finally:
            self.remove_listener(handler)

        if info_cal[0] is True:
            self.calibrated = True
        elif info_cal[0] is False:
            self.calibrated = False

        if info_pos[0] is not None:
            self.position_mm = float(info_pos[0])

        parts = []
        if info_cal[0] is not None:
            parts.append(f"[INFO] Calibrated: {'YES' if info_cal[0] else 'NO'}")
        if info_pos[0] is not None:
            parts.append(f"[INFO] Current position: {self.position_mm:.3f} mm")
        msg = " ".join(parts) if parts else "[INFO] Calibration info read."

        success = (info_cal[0] is True) or (info_pos[0] is not None)
        return LinearAxisResult(success, msg, self.position_mm)

    def _go_to_internal(self, target_mm: float, timeout: float) -> LinearAxisResult:
        if not self.is_connected():
            return LinearAxisResult(False, "Linear axis not connected.", self.position_mm)

        # Clamp but keep floating mm (firmware now accepts fractional mm).
        mm = max(0.0, min(100.0, float(target_mm)))

        done = threading.Event()
        error: List[Optional[str]] = [None]
        last_pos: List[Optional[float]] = [None]

        def handler(msg: str) -> None:
            up = msg.upper()
            if up.startswith("[ERR]") or up.startswith("[CAL ERR]") or up.startswith("[SAFETY]"):
                error[0] = msg
                done.set()
                return
            if "[MOVE] CURRENT POSITION:" in up:
                # Expect: [MOVE] Current position: XX mm
                try:
                    import re

                    m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*MM", up)
                    if m:
                        last_pos[0] = float(m.group(1))
                except Exception:
                    pass
                done.set()
                return
            if "[MOVE] REACHED TARGET POSITION" in up:
                # No position line yet; treat as done but keep waiting for position if it comes.
                done.set()

        self.add_listener(handler)
        try:
            self._emit(f"[AXIS] Goto {mm:.3f} mm...")
            self._send(f"G {mm:.3f}")
            if not done.wait(timeout):
                raise TimeoutError("Move timed out.")
            if error[0]:
                raise RuntimeError(error[0])
        except Exception as ex:
            return LinearAxisResult(False, f"[MOVE] Failed: {ex}", self.position_mm)
        finally:
            self.remove_listener(handler)

        # Update last known position
        if last_pos[0] is not None:
            self.position_mm = float(last_pos[0])
        else:
            self.position_mm = float(mm)
        return LinearAxisResult(True, f"[MOVE] Position {self.position_mm:.1f} mm.", self.position_mm)

    def go_to(self, target_mm: float, timeout: float = 30.0) -> LinearAxisResult:
        """
        Public absolute move; requires calibration to be done first.
        """
        if not self.calibrated:
            return LinearAxisResult(False, "[ERR] Axis not calibrated. Please run calibration first.", self.position_mm)
        return self._go_to_internal(target_mm, timeout)

    def home(self, home_mm: float = 50.0, timeout: float = 30.0) -> LinearAxisResult:
        """
        Move back to the configured home position.
        """
        return self.go_to(home_mm, timeout)


_axis = _LinearAxis()


# Public API used by UI
def refresh_devices() -> List[str]:
    return _axis.list_ports()


def connect(port: str, baud: int = 9600) -> bool:
    return _axis.connect(port, baud)


def disconnect() -> None:
    _axis.disconnect()


def is_connected() -> bool:
    return _axis.is_connected()


def port_name() -> Optional[str]:
    return _axis.port_name()


def is_calibrated() -> bool:
    return _axis.calibrated


def current_position_mm() -> Optional[float]:
    return _axis.position_mm


def add_listener(cb: Callable[[str], None]) -> None:
    _axis.add_listener(cb)


def remove_listener(cb: Callable[[str], None]) -> None:
    _axis.remove_listener(cb)


def calibrate_and_home(home_mm: float = 50.0) -> LinearAxisResult:
    return _axis.calibrate_and_home(home_mm=home_mm)


def read_calibration_info() -> LinearAxisResult:
    return _axis.read_calibration_info()


def goto_mm(target_mm: float) -> LinearAxisResult:
    return _axis.go_to(target_mm)


def home(home_mm: float = 50.0) -> LinearAxisResult:
    return _axis.home(home_mm=home_mm)
