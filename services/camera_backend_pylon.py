"""
Basler (pypylon) camera backend — vendor SDK only.

Uses the pypylon library to talk to Basler cameras. Designed to mirror the
API of `camera_backend_irayple` so `camera_service` can treat both backends
uniformly.

Exposed API (used by services/camera_service):
- available() -> bool
- enumerate_devices() -> List[Dict]
- connect(role, index) -> bool
- disconnect(role) -> None
- is_connected(role) -> bool
- get_connected_index(role) -> Optional[int]
- capture(role, timeout_ms=1500) -> numpy.ndarray (BGR)
- flush(role, frames=2, timeout_ms=50) -> None
- release_all() -> None
- diagnostics() -> Dict
"""

from typing import Dict, List, Optional

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

try:
    from pypylon import pylon  # type: ignore
except Exception as ex:  # pragma: no cover - depends on runtime environment
    pylon = None  # type: ignore
    _import_error = ex
else:
    _import_error = None

_diag: Dict[str, object] = {
    "import_ok": pylon is not None and _import_error is None,
    "load_error": None if _import_error is None else str(_import_error),
    "dev_num": None,
    "last_error": None,
}


_contexts: Dict[str, Dict] = {
    "Top": {"cam": None, "index": None},
    "Front": {"cam": None, "index": None},
}


def available() -> bool:
    """Return True if pypylon and numpy are available."""
    return bool(pylon is not None and np is not None)


def _role_ctx(role: str) -> Dict:
    return _contexts["Top" if str(role) == "Top" else "Front"]


def enumerate_devices(max_indices: int = 8) -> List[Dict]:
    """Enumerate Basler cameras using pypylon."""
    if not available():
        return []
    try:
        tl_factory = pylon.TlFactory.GetInstance()
        devs = tl_factory.EnumerateDevices()
        count = len(devs)
        _diag["dev_num"] = count
        out: List[Dict] = []
        for i, dev in enumerate(devs[: max(0, int(max_indices))]):
            try:
                name = dev.GetFriendlyName()
            except Exception:
                try:
                    name = dev.GetModelName()
                except Exception:
                    try:
                        name = dev.GetSerialNumber()
                    except Exception:
                        name = f"Basler Camera [#{i}]"
            out.append({"index": i, "name": str(name)})
        return out
    except Exception as ex:  # pragma: no cover - hardware/runtime specific
        _diag["last_error"] = str(ex)
        return []


def connect(role: str, index: int) -> bool:
    """Connect the given role to the Basler camera at `index`."""
    if not available():
        return False
    ctx = _role_ctx(role)
    disconnect(role)
    try:
        tl_factory = pylon.TlFactory.GetInstance()
        devs = tl_factory.EnumerateDevices()
        if index < 0 or index >= len(devs):
            return False
        cam = pylon.InstantCamera(tl_factory.CreateDevice(devs[int(index)]))
        cam.Open()
        # Prefer a color/BGR format when possible; ignore failures.
        try:
            if hasattr(cam, "PixelFormat"):
                pf = cam.PixelFormat
                try:
                    symbolics = list(getattr(pf, "GetSymbolics", lambda: [])())
                except Exception:
                    symbolics = []
                for candidate in ("Mono8"):
                    try:
                        if candidate in symbolics:
                            pf.SetValue(candidate)
                            break
                    except Exception:
                        continue
        except Exception:
            pass
        # Grab latest images only to reduce latency.
        cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        ctx["cam"] = cam
        ctx["index"] = int(index)
        _diag["last_error"] = None
        return True
    except Exception as ex:  # pragma: no cover - hardware/runtime specific
        _diag["last_error"] = str(ex)
        return False


def disconnect(role: str) -> None:
    """Disconnect and release the Basler camera for the given role."""
    ctx = _role_ctx(role)
    cam = ctx.get("cam")
    try:
        if cam is None:
            return
        try:
            if cam.IsGrabbing():
                cam.StopGrabbing()
        except Exception:
            pass
        try:
            if cam.IsOpen():
                cam.Close()
        except Exception:
            pass
    except Exception:
        pass
    finally:
        ctx["cam"] = None
        ctx["index"] = None


def is_connected(role: str) -> bool:
    return _role_ctx(role).get("cam") is not None


def get_connected_index(role: str) -> Optional[int]:
    idx = _role_ctx(role).get("index")
    return int(idx) if idx is not None else None


def flush(role: str, frames: int = 2, timeout_ms: int = 50) -> None:
    """Best-effort: drain a few frames from the grab queue."""
    if not available():
        return
    ctx = _role_ctx(role)
    cam = ctx.get("cam")
    if cam is None:
        return
    try:
        if not cam.IsGrabbing():
            cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        for _ in range(max(0, int(frames))):
            try:
                res = cam.RetrieveResult(int(timeout_ms), pylon.TimeoutHandling_ThrowException)
                try:
                    if not res.GrabSucceeded():
                        break
                finally:
                    try:
                        res.Release()
                    except Exception:
                        pass
            except Exception:
                break
    except Exception:
        pass


def capture(role: str, timeout_ms: int = 1500):
    """Capture a single frame as a BGR numpy array."""
    if not available():
        raise RuntimeError("pypylon SDK not available")
    ctx = _role_ctx(role)
    cam = ctx.get("cam")
    if cam is None:
        raise RuntimeError("camera not connected")
    if np is None:
        raise RuntimeError("numpy not available")
    try:
        if not cam.IsGrabbing():
            cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        res = cam.RetrieveResult(int(timeout_ms), pylon.TimeoutHandling_ThrowException)
        if not res.GrabSucceeded():
            raise RuntimeError("Basler grab failed")
        try:
            # Prefer ImageFormatConverter to guarantee BGR8; fallback to raw array.
            arr = None
            try:
                converter = pylon.ImageFormatConverter()
                converter.OutputPixelFormat = getattr(
                    pylon, "PixelType_BGR8packed", getattr(pylon, "PixelType_RGB8packed", None)
                )
                if converter.OutputPixelFormat is not None:
                    converter.OutputBitAlignment = getattr(
                        pylon, "OutputBitAlignment_MsbAligned", converter.OutputBitAlignment
                    )
                image = converter.Convert(res)
                arr = image.GetArray()
            except Exception:
                try:
                    arr = res.Array
                except Exception as ex2:
                    raise RuntimeError(f"Basler frame conversion failed: {ex2}") from ex2
            img = np.asarray(arr)
            if img.ndim == 2:
                img = np.repeat(img[:, :, None], 3, axis=2)
            elif img.ndim == 3 and img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
            # If RGB, approximate BGR by channel swap
            try:
                pf_name = str(cam.PixelFormat.GetValue()).upper()
                if pf_name.startswith("RGB") and img.ndim == 3 and img.shape[2] == 3:
                    img = img[:, :, ::-1]
            except Exception:
                pass
            _diag["last_error"] = None
            return img
        finally:
            try:
                res.Release()
            except Exception:
                pass
    except Exception as ex:  # pragma: no cover - hardware/runtime specific
        _diag["last_error"] = str(ex)
        raise


def release_all() -> None:
    for role in ("Top", "Front"):
        try:
            disconnect(role)
        except Exception:
            pass


def diagnostics() -> Dict:
    return dict(_diag)

