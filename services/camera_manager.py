"""
Centralized camera capture that applies light settings before every shot and
flushes the camera buffer to avoid stale frames. Use this everywhere instead
of calling camera_service.capture() directly.
"""

from typing import Optional
import time
import cv2 as _cv2  # for optional saving

from . import camera_service as _cam
from . import light_controller as _lc
from .config import state as _state


_last_role: Optional[str] = None


def _apply_light(role: str) -> bool:
    """Best-effort: set the configured current on the single light (CH1) based
    on role ('Top' uses Top mA, 'Front' uses Front mA). No revert; each capture
    sets what it needs.
    """
    try:
        st = _state()
        ip = getattr(st, "light_ip", None)
        if not ip:
            return
        _lc.configure(ip)
        # Single light only: always use channel 0 (CH1)
        target = int(
            getattr(
                st,
                "top_current_ma" if str(role).lower() == "top" else "front_current_ma",
                0,
            )
            or 0
        )
        # Toggle sequence handled inside controller: OFF -> set -> ON

        _lc.light_on(0)
        _lc.set_current_toggle(0, target)
        # We did toggle regardless of previous value; signal caller to dwell/flush
        return True
    except Exception:
        # Do not break capture if light is unavailable
        return False


def capture(role: str, *, save_path: Optional[str] = None):
    """Capture a frame from the specified role ('Top' or 'Front').
    Steps:
      1) Apply light (CH1 only) for the role (verified internally by controller).
      2) Flush a couple frames from the backend queue (best-effort).
      3) Capture from the underlying camera_service.
      4) Optionally save to disk.
    """
    # Apply light (read-back ensure is inside light_controller)
    changed = _apply_light(role)

    # Optional dwell only when brightness changed
    try:
        st = _state()
        dwell_ms = int(getattr(st, 'light_dwell_ms', 0) or 0)
    except Exception:
        dwell_ms = 0
    if changed and dwell_ms > 0:
        # Dwell strictly after OFF->SET->ON
        time.sleep(dwell_ms / 1000.0)

    # Flush any buffered frames, then capture
    try:
        from . import camera_service as _svc
        # If brightness changed or role switched, purge deeper to avoid stale frames
        global _last_role
        deep = changed or (_last_role is not None and _last_role != role)
        _svc.flush(role, frames=(4 if deep else 2), timeout_ms=(80 if deep else 50))
        _last_role = role
    except Exception:
        pass

    frame = _cam.capture(role)

    try:
        print(f"[Camera] role={role} captured", flush=True)
    except Exception:
        pass

    if save_path:
        try:
            _cv2.imwrite(str(save_path), frame)
        except Exception:
            pass
    return frame


def is_connected(role: str) -> bool:
    return _cam.is_connected(role)


def connect(role: str, index: int) -> bool:
    return _cam.connect(role, index)


def disconnect(role: str) -> None:
    _cam.disconnect(role)


def enumerate_devices():
    return _cam.enumerate_devices()


def backend_name() -> str:
    return _cam.backend_name()


def diagnostics():
    return _cam.diagnostics()
