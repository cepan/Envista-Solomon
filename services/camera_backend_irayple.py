"""
iRAYPLE (MindVision) camera backend — vendor SDK only.

Uses the iRAYPLE/HuarayTech Python SDK (IMVApi.py/IMVDefines.py) from the
default MV Viewer installation paths. No OpenCV fallback and no environment
variables required.

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
from ctypes import c_uint, c_void_p, byref, cast, c_ubyte
try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

_diag: Dict[str, object] = {
    "import_ok": False,
    "load_error": None,
    "version": None,
    "enum_ret": None,
    "dev_num": None,
    "py_dir": None,
    "runtime_dir": None,
}

import os
import sys


def _try_import_vendor() -> Optional[str]:
    """Attempt to import IMVApi/IMVDefines using default install paths.
    Returns None on success or error string on failure.
    """
    try:
        py_dir = r"C:\\Program Files\\HuarayTech\\MV Viewer\\Development\\Samples\\Python\\IMV\\MVSDK"
        rt_dir = r"C:\\Program Files\\HuarayTech\\MV Viewer\\Runtime\\x64"
        _diag["py_dir"] = py_dir
        _diag["runtime_dir"] = rt_dir
        extra = [
            os.path.join(rt_dir, "GenICam", "bin", "Win64_x64"),
            os.path.join(rt_dir, "GenICam", "bin64"),
            os.path.join(rt_dir, "GenICam", "bin", "win64_x64"),
        ]
        for p in [rt_dir, *extra]:
            if os.path.isdir(p):
                try:
                    if hasattr(os, "add_dll_directory"):
                        os.add_dll_directory(p)  # type: ignore[attr-defined]
                    else:
                        os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
                except Exception:
                    pass
        old_cwd = os.getcwd()
        try:
            if os.path.isdir(py_dir) and py_dir not in sys.path:
                sys.path.insert(0, py_dir)
            if os.path.isdir(py_dir):
                os.chdir(py_dir)
            import IMVApi as _IMVApi  # type: ignore
            import IMVDefines as _IMVDef  # type: ignore
            globals()["IMVApi"] = _IMVApi
            globals()["IMVDef"] = _IMVDef
        finally:
            try:
                os.chdir(old_cwd)
            except Exception:
                pass
        return None
    except Exception as ex:
        return str(ex)


# Try vendor SDK now (single path). No fallback.
IMVApi = None
IMVDef = None
err = _try_import_vendor()
if err is None:
    _diag["import_ok"] = True
    # Optional: version will be filled later in available() if exposed
else:
    _diag["load_error"] = err


_contexts: Dict[str, Dict] = {
    "Top": {"cam": None, "index": None},
    "Front": {"cam": None, "index": None},
}


def available() -> bool:
    return bool(IMVApi is not None and IMVDef is not None and np is not None)


def enumerate_devices(max_indices: int = 8) -> List[Dict]:
    if IMVApi is None or IMVDef is None:
        return []
    try:
        dev_list = IMVDef.IMV_DeviceList()
        interface_all = getattr(IMVDef.IMV_EInterfaceType, "interfaceTypeAll", 0x0F)
        ret = IMVApi.MvCamera.IMV_EnumDevices(dev_list, int(interface_all))
        _diag["enum_ret"] = ret
        try:
            count = int(getattr(dev_list, "nDevNum", 0) or 0)
        except Exception:
            count = 0
        if count <= 0:
            return []

        def _dec(field) -> Optional[str]:
            try:
                if field is None:
                    return None
                b = None
                if hasattr(field, 'raw'):
                    b = field.raw
                elif hasattr(field, 'value'):
                    b = field.value  # type: ignore[attr-defined]
                else:
                    b = bytes(field)
                s = b.split(b'\x00', 1)[0].decode(errors='ignore').strip()
                return s or None
            except Exception:
                return None

        names_seen = set()
        out: List[Dict] = []
        for i in range(count):
            try:
                info = dev_list.pDevInfo[i]  # type: ignore[index]
                model = _dec(getattr(info, 'modelName', None))
                camera_name = _dec(getattr(info, 'cameraName', None))
                vendor = _dec(getattr(info, 'vendorName', None))
                serial = _dec(getattr(info, 'serialNumber', None))
                iface = _dec(getattr(info, 'interfaceName', None))
                # IP for GigE if present
                ip = None
                try:
                    ip = _dec(info.DeviceSpecificInfo.gigeDeviceInfo.ipAddress)
                except Exception:
                    ip = None
                base = camera_name or model or vendor or 'Camera'
                parts = []
                if serial:
                    parts.append(serial)
                if ip:
                    parts.append(ip)
                if iface:
                    parts.append(iface)
                suffix = f" ({', '.join(parts)})" if parts else ""
                name = f"{base}{suffix} [#{i}]"
            except Exception:
                name = f"Camera [#{i}]"
            if name in names_seen:
                name = f"{name}*"
            names_seen.add(name)
            out.append({"index": i, "name": name})
        return out
    except Exception:
        return []


def _role_ctx(role: str) -> Dict:
    return _contexts["Top" if str(role) == "Top" else "Front"]


def connect(role: str, index: int) -> bool:
    if not available():
        return False
    ctx = _role_ctx(role)
    disconnect(role)
    try:
        cam = IMVApi.MvCamera()
        idx = c_uint(int(index))
        mode = int(getattr(IMVDef.IMV_ECreateHandleMode, 'modeByIndex', 0))
        cam.IMV_CreateHandle(mode, cast(byref(idx), c_void_p))
        ret = cam.IMV_Open()
        if ret is not None and ret != getattr(IMVDef, 'IMV_OK', 0):
            return False
        try:
            cam.IMV_StartGrabbing()
        except Exception:
            pass
        ctx['cam'] = cam
        ctx['index'] = int(index)
        return True
    except Exception:
        return False


def disconnect(role: str) -> None:
    ctx = _role_ctx(role)
    cam = ctx.get("cam")
    try:
        if cam is None:
            return
        # Vendor instance has IMV_StopGrabbing/IMV_Close
        if IMVApi is not None and hasattr(cam, "IMV_StopGrabbing"):
            try:
                cam.IMV_StopGrabbing()
            except Exception:
                pass
            try:
                cam.IMV_Close()
            except Exception:
                pass
            try:
                cam.IMV_DestroyHandle()
            except Exception:
                pass
        else:
            # Unknown object; ignore
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
    cam = _role_ctx(role).get("cam")
    if cam is None:
        return
    try:
        try:
            cam.IMV_ClearFrameBuffer()
        except Exception:
            pass
        for _ in range(max(0, int(frames))):
            frm = IMVDef.IMV_Frame()
            try:
                ret = cam.IMV_GetFrame(frm, int(timeout_ms))
                if ret != getattr(IMVDef, 'IMV_OK', 0):
                    break
            finally:
                try:
                    cam.IMV_ReleaseFrame(frm)
                except Exception:
                    pass
    except Exception:
        pass


def capture(role: str, timeout_ms: int = 1500):
    cam = _role_ctx(role).get("cam")
    if cam is None:
        raise RuntimeError("camera not connected")
    if IMVApi is None or np is None:
        raise RuntimeError("vendor SDK not available")
    frm = IMVDef.IMV_Frame()
    ret = cam.IMV_GetFrame(frm, int(timeout_ms))
    if ret != getattr(IMVDef, 'IMV_OK', 0):
        raise RuntimeError(f"IMV_GetFrame failed: {ret}")
    try:
        w = int(frm.frameInfo.width)
        h = int(frm.frameInfo.height)
        size = int(frm.frameInfo.size)
        pix = int(frm.frameInfo.pixelFormat)
        if pix == int(getattr(IMVDef.IMV_EPixelType, 'gvspPixelBGR8', -1)):
            src = np.ctypeslib.as_array(frm.pData, shape=(size,))
            img = src.reshape(h, w, 3).copy()
        else:
            dst_size = w * h * 3
            dst_buf = (c_ubyte * dst_size)()
            pc = IMVDef.IMV_PixelConvertParam()
            pc.nWidth = w
            pc.nHeight = h
            pc.ePixelFormat = pix
            pc.pSrcData = frm.pData
            pc.nSrcDataLen = size
            pc.nPaddingX = int(frm.frameInfo.paddingX)
            pc.nPaddingY = int(frm.frameInfo.paddingY)
            pc.eBayerDemosaic = int(getattr(IMVDef.IMV_EBayerDemosaic, 'demosaicBilinear', 1))
            pc.eDstPixelFormat = int(getattr(IMVDef.IMV_EPixelType, 'gvspPixelBGR8', 0))
            pc.pDstBuf = dst_buf
            pc.nDstBufSize = dst_size
            ret2 = cam.IMV_PixelConvert(pc)
            if ret2 != getattr(IMVDef, 'IMV_OK', 0):
                # fallback mono
                src = np.ctypeslib.as_array(frm.pData, shape=(size,))
                if size >= w * h:
                    gray = src[: w * h].reshape(h, w)
                    img = np.repeat(gray[:, :, None], 3, axis=2).copy()
                else:
                    raise RuntimeError(f"pixel convert failed: {ret2}")
            else:
                out_len = int(getattr(pc, 'nDstDataLen', dst_size) or dst_size)
                dst = np.ctypeslib.as_array(dst_buf, shape=(out_len,))
                img = dst.reshape(h, w, 3).copy()
        return img
    finally:
        try:
            cam.IMV_ReleaseFrame(frm)
        except Exception:
            pass


def release_all():
    for role in ("Top", "Front"):
        try:
            disconnect(role)
        except Exception:
            pass


def diagnostics() -> Dict:
    return dict(_diag)
