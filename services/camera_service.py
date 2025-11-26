"""
Multi-backend camera service.

Supports multiple vendor backends (iRAYPLE, Basler/pypylon, etc.) and exposes
a unified API to the rest of the app.

API:
- enumerate_devices() -> list of {index:int, name:str, backend:str, backend_index:int}
- connect(role:str, index:int) -> bool            # index is global (across backends)
- disconnect(role:str) -> None
- is_connected(role:str) -> bool
- get_connected_index(role:str) -> Optional[int]  # returns global index
- capture(role:str) -> numpy.ndarray (BGR)
- flush(role:str, frames:int=2, timeout_ms:int=50) -> None
- release_all() -> None
- backend_name() -> str
- diagnostics() -> dict
"""

from typing import Dict, List, Optional, Tuple

try:
    from . import camera_backend_irayple as _irayple
except Exception:
    _irayple = None

try:
    from . import camera_backend_pylon as _pylon
except Exception:
    _pylon = None

try:
    import cv2 as _cv2  # type: ignore
except Exception:
    _cv2 = None  # type: ignore


# Global device table: each entry has a unique global index per process.
_DEVICES: List[Dict[str, object]] = []

# Per-role connection context: maps "Top"/"Front" to backend + indices.
_ROLE_CONN: Dict[str, Optional[Dict[str, object]]] = {"Top": None, "Front": None}


def _normalize_role(role: str) -> str:
    return "Top" if str(role) == "Top" else "Front"


def _available_backends() -> List[Tuple[str, object]]:
    backends: List[Tuple[str, object]] = []
    if _irayple is not None and getattr(_irayple, "available", None):
        try:
            if _irayple.available():
                backends.append(("iRAYPLE", _irayple))
        except Exception:
            pass
    if _pylon is not None and getattr(_pylon, "available", None):
        try:
            if _pylon.available():
                backends.append(("PYLON", _pylon))
        except Exception:
            pass
    return backends


def backend_name() -> str:
    """Human-readable summary of active backends."""
    names = [name for name, _ in _available_backends()]
    if not names:
        return "None"
    if len(names) == 1:
        return names[0]
    return "+".join(names)


def _rebuild_device_table(max_devices_per_backend: int = 8) -> None:
    """Enumerate devices from all backends and assign global indices."""
    global _DEVICES
    devices: List[Dict[str, object]] = []
    next_index = 0
    for name, backend in _available_backends():
        try:
            enum = getattr(backend, "enumerate_devices", None)
            if enum is None:
                continue
            backend_devs = enum(max_devices_per_backend) or []
        except Exception:
            backend_devs = []
        for d in backend_devs:
            try:
                local_idx = int(d.get("index", 0))
            except Exception:
                local_idx = 0
            label = str(d.get("name", f"Camera {local_idx}"))
            # Append backend tag to name so users can distinguish.
            name_with_backend = f"{label} [{name}]"
            devices.append(
                {
                    "index": next_index,  # global index
                    "name": name_with_backend,
                    "backend": name,
                    "backend_index": local_idx,
                }
            )
            next_index += 1
    _DEVICES = devices


def _ensure_device_table() -> None:
    if not _DEVICES:
        _rebuild_device_table()


def enumerate_devices(max_devices: int = 8) -> List[Dict]:
    """Return combined device list across all available backends."""
    _rebuild_device_table(max_devices)
    # Expose a shallow copy so callers don't mutate internal state.
    return list(_DEVICES)


def _find_device(global_index: int) -> Optional[Dict[str, object]]:
    """Lookup a device by global index; rebuild table once if needed."""
    _ensure_device_table()
    for d in _DEVICES:
        try:
            if int(d.get("index", -1)) == int(global_index):
                return d
        except Exception:
            continue
    # Retry after a rebuild in case devices changed.
    _rebuild_device_table()
    for d in _DEVICES:
        try:
            if int(d.get("index", -1)) == int(global_index):
                return d
        except Exception:
            continue
    return None


def _backend_module(name: str):
    if name == "iRAYPLE":
        return _irayple
    if name == "PYLON":
        return _pylon
    return None


def connect(role: str, index: int) -> bool:
    """Connect a role ('Top'/'Front') to a device by global index."""
    role = _normalize_role(role)
    dev = _find_device(index)
    if not dev:
        return False
    backend_name_str = str(dev.get("backend") or "")
    backend_idx = int(dev.get("backend_index", 0) or 0)
    backend = _backend_module(backend_name_str)
    if backend is None or not getattr(backend, "available", lambda: False)():
        return False
    # Disconnect any previous camera for this role.
    disconnect(role)
    try:
        ok = backend.connect(role, backend_idx)
    except Exception:
        ok = False
    if ok:
        _ROLE_CONN[role] = {
            "index": int(dev["index"]),
            "backend": backend_name_str,
            "backend_index": backend_idx,
        }
    return ok


def disconnect(role: str) -> None:
    role = _normalize_role(role)
    ctx = _ROLE_CONN.get(role)
    if ctx:
        backend_name_str = str(ctx.get("backend") or "")
        backend = _backend_module(backend_name_str)
        if backend is not None and getattr(backend, "disconnect", None):
            try:
                backend.disconnect(role)
            except Exception:
                pass
    _ROLE_CONN[role] = None


def is_connected(role: str) -> bool:
    role = _normalize_role(role)
    ctx = _ROLE_CONN.get(role)
    if not ctx:
        return False
    backend_name_str = str(ctx.get("backend") or "")
    backend = _backend_module(backend_name_str)
    if backend is None or not getattr(backend, "is_connected", None):
        return False
    try:
        return bool(backend.is_connected(role))
    except Exception:
        return False


def get_connected_index(role: str) -> Optional[int]:
    role = _normalize_role(role)
    ctx = _ROLE_CONN.get(role)
    if not ctx:
        return None
    try:
        return int(ctx.get("index"))  # global index
    except Exception:
        return None


def _role_backend(role: str):
    role = _normalize_role(role)
    ctx = _ROLE_CONN.get(role)
    if not ctx:
        return None, None
    backend_name_str = str(ctx.get("backend") or "")
    backend = _backend_module(backend_name_str)
    return backend, role


def capture(role: str):
    """Capture a frame from the specified role using its bound backend."""
    backend, role_norm = _role_backend(role)
    if backend is None:
        raise RuntimeError("camera not connected")
    if not getattr(backend, "available", lambda: False)():
        raise RuntimeError("camera backend not available")
    try:
        frame = backend.capture(role_norm)
        # Hard-coded orientation adjustment: rotate Top camera 90° clockwise
        if _normalize_role(role) == "Top" and _cv2 is not None:
            try:
                frame = _cv2.rotate(frame, _cv2.ROTATE_90_CLOCKWISE)
            except Exception:
                # If rotation fails, fall back to original frame
                pass
        return frame
    except Exception:
        # Re-raise to caller; message comes from backend where possible.
        raise


def flush(role: str, frames: int = 2, timeout_ms: int = 50) -> None:
    backend, role_norm = _role_backend(role)
    if backend is None or not getattr(backend, "available", lambda: False)():
        return
    fn = getattr(backend, "flush", None)
    if fn is None:
        return
    try:
        fn(role_norm, frames, timeout_ms)
    except Exception:
        pass


def release_all() -> None:
    # Ask each backend to release its own resources.
    for name, backend in _available_backends():
        fn = getattr(backend, "release_all", None)
        if fn is not None:
            try:
                fn()
            except Exception:
                pass
    for role in ("Top", "Front"):
        _ROLE_CONN[role] = None


def diagnostics() -> Dict:
    """Return diagnostics for all backends and the current device table."""
    info: Dict[str, object] = {}
    backends: List[str] = []
    if _irayple is not None:
        backends.append("iRAYPLE")
        try:
            info["iRAYPLE"] = _irayple.diagnostics()
        except Exception as ex:
            info["iRAYPLE"] = {"error": str(ex)}
    if _pylon is not None:
        backends.append("PYLON")
        try:
            info["PYLON"] = _pylon.diagnostics()
        except Exception as ex:
            info["PYLON"] = {"error": str(ex)}
    info["backends"] = backends
    info["devices"] = list(_DEVICES)
    return info
