from dataclasses import dataclass, asdict, fields
from typing import Optional, Any
from pathlib import Path
import json


@dataclass
class InitSettings:
    attachment_path: Optional[str] = None
    front_attachment_path: Optional[str] = None
    defect_path: Optional[str] = None
    top_preview_np: Optional[Any] = None
    front_preview_np: Optional[Any] = None


_settings = InitSettings()


def settings() -> InitSettings:
    return _settings


# Persisted app state (lightweight UI state only; no images persisted)
@dataclass
class AppState:
    last_project_path: Optional[str] = None
    last_image_path: Optional[str] = None
    attachment_path: Optional[str] = None
    front_attachment_path: Optional[str] = None
    defect_path: Optional[str] = None
    camera_top_index: Optional[int] = None
    camera_front_index: Optional[int] = None
    turntable_port: Optional[str] = None
    turntable_step: Optional[float] = None
    linear_axis_port: Optional[str] = None
    linear_axis_last_mm: Optional[float] = None
    linear_axis_home_mm: Optional[float] = None
    overlay_enabled: Optional[bool] = None
    # Horizontal FOV of the front camera as measured in the top camera image (pixels)
    front_fov_top_px: Optional[float] = None
    solvision_score_threshold: Optional[float] = None
    solvision_nms_threshold: Optional[float] = None
    solvision_max_detections: Optional[int] = None
    # Persisted contour/edge tuning parameters for arrow computation
    contour_params: Optional[dict] = None
    # Step-2 square crop size (pixels)
    step2_crop_size: Optional[int] = None
    # Light controller settings
    light_ip: Optional[str] = None
    light_enabled: Optional[bool] = None
    top_current_ma: Optional[int] = None
    front_current_ma: Optional[int] = None
    # Dwell time after setting brightness (ms). If set, capture waits this
    # long only when brightness is changed for the capture role.
    light_dwell_ms: Optional[int] = None
    # Part identifier for captures
    part_id: Optional[str] = None


_state = AppState()


def _state_path() -> Path:
    # Store next to python_export as user_settings.json
    base = Path(__file__).resolve().parents[1]
    return base / "user_settings.json"


def load_state() -> AppState:
    global _state
    p = _state_path()
    try:
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            # Ignore unknown keys to remain forward/backward compatible
            allowed = {f.name for f in fields(AppState)}
            filtered = {k: v for k, v in (data or {}).items() if k in allowed}
            _state = AppState(**filtered)
    except Exception:
        # Corrupt or incompatible; start fresh
        _state = AppState()
    return _state


def save_state() -> None:
    p = _state_path()
    try:
        p.write_text(json.dumps(asdict(_state), ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def state() -> AppState:
    return _state


# Load on import
load_state()
