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
class TurntableHomeResult:
    success: bool
    message: str
    offset_degrees: Optional[float] = None


class _Turntable:
    def __init__(self):
        self._ser: Optional[serial.Serial] = None if serial else None
        self._lock = threading.Lock()
        self._guard_ms = 100
        self._last_cmd_ts = 0.0
        self._recv_buffer = ""
        self._reader: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._listeners: List[Callable[[str], None]] = []
        self.is_homed = False
        self.last_offset_angle: Optional[float] = None

    # Discovery
    def list_ports(self) -> List[str]:
        if list_ports is None:
            return []
        return [p.device for p in list_ports.comports()]

    # Connection
    def connect(self, port: str, baud: int = 115200) -> bool:
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
            self.is_homed = False
            self.last_offset_angle = None
            self._start_reader()
            self._emit(f"INFO Connected to {port}")
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
        self.is_homed = False
        self.last_offset_angle = None

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

    def _start_reader(self):
        if self._reader and self._reader.is_alive():
            return
        self._stop.clear()
        self._reader = threading.Thread(target=self._reader_loop, name="tt_reader", daemon=True)
        self._reader.start()

    def _stop_reader(self):
        self._stop.set()
        if self._reader and self._reader.is_alive():
            self._reader.join(timeout=0.5)
        self._reader = None

    def _reader_loop(self):
        ser = self._ser
        if ser is None:
            return
        while not self._stop.is_set():
            try:
                chunk = ser.read(1024)
                if not chunk:
                    continue
                text = chunk.decode("ascii", errors="ignore")
                self._recv_buffer += text
                while ";" in self._recv_buffer:
                    msg, self._recv_buffer = self._recv_buffer.split(";", 1)
                    norm = self.normalize(msg)
                    if norm:
                        self._emit(norm)
            except Exception:
                time.sleep(0.05)

    @staticmethod
    def normalize(message: str) -> str:
        if not message:
            return ""
        m = message.strip().lstrip('#').strip()
        return m

    def _send_command(self, cmd: str) -> None:
        if not self.is_connected():
            raise RuntimeError("Turntable not connected.")
        now = time.time() * 1000.0
        with self._lock:
            delta = now - self._last_cmd_ts
            if delta < self._guard_ms:
                time.sleep((self._guard_ms - delta) / 1000.0)
            payload = cmd if cmd.endswith(";") else cmd + ";"
            self._ser.write(payload.encode("ascii"))
            self._ser.flush()
            self._last_cmd_ts = time.time() * 1000.0

    # High-level ops
    def request_offset(self, timeout: float = 2.0) -> float:
        if not self.is_connected():
            raise RuntimeError("Turntable not connected.")
        done = threading.Event()
        result: List[Optional[float]] = [None]
        error: List[Optional[str]] = [None]

        def handler(msg: str):
            if msg.upper().startswith("CR+ERR"):
                error[0] = msg
                done.set()
                return
            if "OffsetAngle" in msg:
                # Expect format like: ... OffsetAngle=xx.xx ...
                import re
                m = re.search(r"-?\d+(?:\.\d+)?", msg)
                if m:
                    try:
                        result[0] = float(m.group(0))
                        done.set()
                    except Exception:
                        error[0] = "Parse error"
                        done.set()

        self.add_listener(handler)
        try:
            self._send_command("CT+GETOFFSETANGLE()")
            if not done.wait(timeout):
                raise TimeoutError("Offset request timed out")
            if error[0]:
                raise RuntimeError(error[0])
            if result[0] is None:
                raise RuntimeError("Offset not received")
            return float(result[0])
        finally:
            self.remove_listener(handler)

    def home(self, timeout: float = 20.0) -> TurntableHomeResult:
        if not self.is_connected():
            return TurntableHomeResult(False, "Turntable not connected.")
        self.is_homed = False
        self.last_offset_angle = None

        try:
            offset = self.request_offset(timeout=2.0)
            self.last_offset_angle = offset
            move_angle = abs(offset)
            if move_angle < 1e-3 or abs(move_angle - 360.0) < 1e-3:
                self.is_homed = True
                self.last_offset_angle = 0.0
                return TurntableHomeResult(True, "Homing complete (already at zero).", offset)

            direction = 1 if offset > 0 else 0  # per C# homing logic: 1=CCW, 0=CW
            direction_label = "CCW" if direction == 1 else "CW"

            ack = threading.Event()
            done = threading.Event()
            error: List[Optional[str]] = [None]

            def handler(msg: str):
                up = msg.upper()
                if up.startswith("CR+ERR"):
                    error[0] = msg
                    ack.set(); done.set()
                    return
                if not ack.is_set() and up.startswith("CR+OK"):
                    ack.set()
                    return
                if up.startswith("CR+EVENT=TB_END"):
                    done.set()

            self.add_listener(handler)
            try:
                cmd = f"CT+START({direction},1,0,{move_angle:.4f},0,1)"
                self._send_command(cmd)
                if not ack.wait(timeout):
                    raise TimeoutError("Homing ack timed out")
                if not done.wait(timeout):
                    raise TimeoutError("Homing completion timed out")
                if error[0]:
                    raise RuntimeError(error[0])
                self.is_homed = True
                self.last_offset_angle = 0.0
                return TurntableHomeResult(True, f"Homing complete. Rotated {move_angle:.2f} deg {direction_label} to zero.", offset)
            finally:
                self.remove_listener(handler)
        except Exception as ex:
            self.is_homed = False
            return TurntableHomeResult(False, str(ex))

    def move_relative(self, angle_deg: float, timeout: float = 10.0) -> str:
        if not self.is_connected():
            raise RuntimeError("Turntable not connected.")
        mag = abs(angle_deg)
        if mag < 1e-3:
            return "Rotation skipped (below threshold)."
        direction = 0 if angle_deg >= 0 else 1  # 0=CW, 1=CCW
        direction_label = "CW" if direction == 0 else "CCW"
        ack = threading.Event()
        done = threading.Event()
        error: List[Optional[str]] = [None]

        def handler(msg: str):
            up = msg.upper()
            if up.startswith("CR+ERR"):
                error[0] = msg
                ack.set(); done.set()
                return
            if not ack.is_set() and up.startswith("CR+OK"):
                ack.set(); return
            if up.startswith("CR+EVENT=TB_END"):
                done.set()

        self.add_listener(handler)
        try:
            cmd = f"CT+START({direction},1,0,{mag:.4f},0,1)"
            self._send_command(cmd)
            if not ack.wait(timeout):
                raise TimeoutError("Rotation ack timed out")
            if not done.wait(timeout):
                raise TimeoutError("Rotation completion timed out")
            if error[0]:
                raise RuntimeError(error[0])
            return f"Rotated {mag:.2f} deg {direction_label}."
        finally:
            self.remove_listener(handler)


_tt = _Turntable()


# Public API used by UI
def refresh_devices() -> List[str]:
    return _tt.list_ports()


def connect(port: str, baud: int = 115200) -> bool:
    return _tt.connect(port, baud)


def disconnect() -> None:
    _tt.disconnect()


def is_connected() -> bool:
    return _tt.is_connected()


def port_name() -> Optional[str]:
    return _tt.port_name()


def add_listener(cb: Callable[[str], None]) -> None:
    _tt.add_listener(cb)


def remove_listener(cb: Callable[[str], None]) -> None:
    _tt.remove_listener(cb)


def home() -> TurntableHomeResult:
    return _tt.home()


def move_relative(angle_deg: float) -> str:
    return _tt.move_relative(angle_deg)
