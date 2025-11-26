import socket
import struct
import time
from typing import Optional

# ULC-2 UDP protocol constants
CMD_OUTPUT_READ        = 0xDE
CMD_OUTPUT_SET         = 0xDC  # data: 01 = enable, 00 = disable
CMD_MODE_READ          = 0xE1
CMD_MODE_SET           = 0xDF  # data: 03 = DC, 04 = Strobe, 05 = Voltage
CMD_DC_AMPS_READ       = 0xB6
CMD_DC_AMPS_SET        = 0xB7
CMD_MAX_DC_AMPS_READ   = 0xBA
CMD_MAX_DC_AMPS_SET    = 0xBB

DEFAULT_PORT = 5000

_sock: Optional[socket.socket] = None
_ip: Optional[str] = None
_port: int = DEFAULT_PORT
_enabled: bool = True
_last_currents = {0: 0, 1: 0}


def _dprint(*args):
    try:
        print("[Light]", *args, flush=True)
    except Exception:
        pass


def _build_packet(channel: int, command_byte: int, data_24bit: int) -> bytes:
    length = 8
    reserved = 0
    return struct.pack(
        ">HBBB", length, channel & 0xFF, command_byte & 0xFF, reserved
    ) + int(data_24bit & 0xFFFFFF).to_bytes(3, "big")


def configure(ip: str, *, port: int = DEFAULT_PORT, enabled: bool = True) -> None:
    """Configure controller destination and ensure socket exists."""
    global _sock, _ip, _port, _enabled
    _ip = (ip or "").strip() or None
    _port = int(port) if port else DEFAULT_PORT
    _enabled = bool(enabled)
    try:
        if _sock is None:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.15)  # a bit longer to tolerate device latency
            _sock = s
    except Exception:
        _sock = None


def _send(channel: int, cmd: int, data24: int) -> None:
    if _sock is None or not _ip:
        return
    try:
        pkt = _build_packet(channel, cmd, data24)
        _sock.sendto(pkt, (_ip, _port))
        # Keep logs concise; DC_AMPS_SET can be frequent during ensure loops
        if cmd in (CMD_MODE_SET, CMD_MAX_DC_AMPS_SET, CMD_OUTPUT_SET):
            _dprint(f"send ch={channel} cmd=0x{cmd:02X} data={data24} to {_ip}:{_port}")
    except Exception:
        pass


def _read_dc_current(channel: int) -> Optional[int]:
    """Read back DC current (mA) for channel; None on failure/timeouts."""
    if _sock is None or not _ip:
        return None
    try:
        pkt = _build_packet(channel, CMD_DC_AMPS_READ, 0)
        _sock.sendto(pkt, (_ip, _port))
        data, _ = _sock.recvfrom(64)
        # Some devices may echo channel differently; trust the command byte
        if len(data) >= 8 and data[3] == CMD_DC_AMPS_READ:
            return int.from_bytes(data[5:8], "big", signed=False)
    except Exception:
        return None
    return None


def light_on(channel: int) -> None:
    _send(channel, CMD_MODE_SET, 0x000003)  # DC mode
    _send(channel, CMD_OUTPUT_SET, 0x000001)  # enable


def light_off(channel: int) -> None:
    _send(channel, CMD_OUTPUT_SET, 0x000000)


def set_current(channel: int, milliamps: int) -> None:
    """Set DC current and verify by read-back (no output toggle)."""
    if channel not in (0, 1):
        return
    try:
        target = max(0, min(4000, int(milliamps)))
    except Exception:
        target = 0
    _last_currents[channel] = target
    if not _enabled:
        return
    # Allow target current and attempt a short verify loop
    _send(channel, CMD_MAX_DC_AMPS_SET, 250)
    deadline = time.perf_counter() + 0.30
    fb = None
    tries = 0
    while time.perf_counter() < deadline and tries < 6:
        _send(channel, CMD_DC_AMPS_SET, target)
        time.sleep(0.02)
        fb = _read_dc_current(channel)
        if fb == target:
            break
        tries += 1
    if fb != target:
        _dprint(f"ensure mismatch: target={target} read={fb}")


def set_current_toggle(channel: int, milliamps: int) -> None:
    """Safely switch brightness: OFF -> set current -> ON, with verify.

    This sequence minimizes visible flicker when changing brightness mid-stream
    and matches the desired hardware behavior.
    """
    if channel not in (0, 1):
        return
    try:
        target = max(0, min(4000, int(milliamps)))
    except Exception:
        target = 0
    _last_currents[channel] = target
    if not _enabled:
        return
    # Helper to attempt setting/reading on a specific channel
    def _attempt(ch: int) -> Optional[int]:
        # Pre-stage limits and current while off
        _send(ch, CMD_DC_AMPS_SET, target)
        # Verify quickly after enabling; if mismatch, retry set a few times
        deadline = time.perf_counter() + 0.30
        fb_local = None
        tries = 0
        while time.perf_counter() < deadline and tries < 6:
            time.sleep(0.02)
            fb_local = _read_dc_current(ch)
            if fb_local == target:
                return fb_local
            _send(ch, CMD_DC_AMPS_SET, target)
            tries += 1
        return fb_local

    # First try requested channel
    fb = _attempt(channel)
    if fb != target:
        # Some ULC-2 models map DC amps to the alternate channel index; try it once
        alt = 1 - (channel & 1)
        fb = _attempt(alt)
    if fb != target:
        _dprint(f"ensure mismatch after toggle: target={target} read={fb}")


def get_current(channel: int) -> int:
    # Prefer actual read-back when possible
    fb = _read_dc_current(channel)
    if isinstance(fb, int):
        _last_currents[channel] = fb
        return fb
    return int(_last_currents.get(channel, 0))


def apply_for_role(role: str, top_ma: Optional[int], front_ma: Optional[int]) -> None:
    """Single-light (CH1) convenience: always drive channel 0 regardless of role.
    Top uses top_ma; Front uses front_ma to select the target value only.
    """
    ch = 0
    target_raw = top_ma if str(role).lower() == 'top' else front_ma
    try:
        target = int(target_raw) if target_raw is not None else 0
    except Exception:
        target = 0
    set_current_toggle(ch, target)
    _dprint(f"apply role={role} ch={ch} target={target}mA")
