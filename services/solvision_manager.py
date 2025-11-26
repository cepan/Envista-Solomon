"""
Detectron2-backed model loader and inference helper.

Provides the same public function names the UI expects (load_project,
detect, detect_for, etc.) without any SolVision/.NET dependencies.
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

from .config import state as _state

_predictors: Dict[str, DefaultPredictor] = {}  # e.g., {"top": pred, "front": pred2}
_model_paths: Dict[str, str] = {}
_initialized_error: Optional[str] = None
_log_cb: Optional[Callable[[str], None]] = None

# Default thresholds and class list
_DEFAULT_SCORE_THRESHOLD = 0.4
# Higher default for top (step 1) detections
_DEFAULT_SCORE_THRESHOLD_TOP = 0.9
CLASS_NAMES: List[str] = ["attachment"]
# Per-model class names; defaults applied when loading.
_class_names_per_model: Dict[str, List[str]] = {
    "top": CLASS_NAMES,
    "front": CLASS_NAMES,
    "defect": ["defect"],
}


def set_ui_logger(cb: Optional[Callable[[str], None]]):
    """Optional UI logger callback."""
    global _log_cb
    _log_cb = cb


def _dprint(*args):
    try:
        msg = " ".join(str(a) for a in args)
        print("[Detectron]", msg, flush=True)
        if _log_cb is not None:
            try:
                _log_cb("[Detectron] " + msg)
            except Exception:
                pass
    except Exception:
        pass


def _build_predictor(model_path: str, class_names: List[str]) -> DefaultPredictor:
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)
    cfg.MODEL.WEIGHTS = model_path
    # Keep detectron threshold low; we filter manually per request
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DefaultPredictor(cfg)


def ensure_initialized() -> bool:
    """Detectron pipeline has no global init; always ready."""
    global _initialized_error
    _initialized_error = None
    return True


def initialization_error() -> Optional[str]:
    return _initialized_error


def _coerce_float(val, default: float) -> float:
    try:
        if val is None:
            return default
        return float(val)
    except Exception:
        return default


def load_project(path: str) -> None:
    """Load the default (top) model."""
    load_project_for("top", path)


def load_project_inproc(path: str) -> None:
    # Kept for backward compatibility with worker scripts (no-op difference).
    load_project(path)


def load_project_for(name: str, path: str, *, mode: str = "exe") -> None:
    """Load a model checkpoint for a given role (top/front/defect)."""
    global _initialized_error
    if not path:
        raise ValueError("Empty model path")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    try:
        class_names = _class_names_per_model.get(name, CLASS_NAMES if name != "defect" else ["defect"])
        if name not in _class_names_per_model:
            _class_names_per_model[name] = class_names
        pred = _build_predictor(path, class_names)
        _predictors[name] = pred
        _model_paths[name] = os.path.abspath(path)
        if name == "top":
            # Mirror legacy single-project tracking
            _model_paths["_default"] = _model_paths[name]
        _dprint(f"Loaded model for '{name}': {path}")
    except Exception as exc:
        _initialized_error = str(exc)
        _dprint(f"Failed to load model '{name}': {exc}")
        raise


def has_loaded_project() -> bool:
    return "top" in _predictors


def current_project_path() -> Optional[str]:
    return _model_paths.get("top")


def current_project_path_for(name: str) -> Optional[str]:
    return _model_paths.get(name)


def diagnostics() -> Dict[str, Any]:
    return {
        "loaded": list(_predictors.keys()),
        "models": dict(_model_paths),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "initialized": True,
        "error": _initialized_error,
    }


def diagnostics_text() -> str:
    d = diagnostics()
    lines = [
        f"loaded: {d.get('loaded')}",
        f"models: {d.get('models')}",
        f"device: {d.get('device')}",
    ]
    if d.get("error"):
        lines.append(f"error: {d.get('error')}")
    return "\n".join(lines)


def _normalize_detections(instances, score_threshold: float, class_names: List[str]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    if instances is None:
        return results

    boxes = instances.pred_boxes if instances.has("pred_boxes") else None
    scores = instances.scores if instances.has("scores") else None
    classes = instances.pred_classes if instances.has("pred_classes") else None
    masks = instances.pred_masks if instances.has("pred_masks") else None

    num = len(instances)
    for i in range(num):
        sc = float(scores[i]) if scores is not None else 0.0
        if sc < score_threshold:
            continue
        box = boxes[i].tensor.numpy().tolist()[0] if boxes is not None else None
        if box is None:
            continue
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        cls_id = int(classes[i]) if classes is not None else None
        cls_name = (
            class_names[cls_id]
            if cls_id is not None and 0 <= cls_id < len(class_names)
            else str(cls_id)
        )
        mask = masks[i].numpy() if masks is not None else None
        area = float(w * h) if w is not None and h is not None else None
        if mask is not None:
            try:
                area = float(mask.sum())
            except Exception:
                pass
        results.append(
            {
                "class": cls_name,
                "class_id": cls_id,
                "score": sc,
                "bounds": (float(x1), float(y1), float(w), float(h)),
                "area": area,
                "mask": mask,
                "rect": {"x": float(x1), "y": float(y1), "width": float(w), "height": float(h)},
            }
        )
    return results


def detect(image_path: str, score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
    """Run detection with the default (top) model."""
    return detect_for("top", image_path, score_threshold=score_threshold)


def detect_inproc(image_path: str, score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
    # Kept for compatibility; identical to detect().
    return detect(image_path, score_threshold=score_threshold)


def detect_for(name: str, image_path: str, score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
    if not image_path:
        raise ValueError("Empty image path")
    if name not in _predictors:
        raise RuntimeError(f"Model '{name}' not loaded")

    default_thr = _DEFAULT_SCORE_THRESHOLD_TOP if name == "top" else _DEFAULT_SCORE_THRESHOLD
    thr = _coerce_float(
        score_threshold,
        _coerce_float(getattr(_state(), "solvision_score_threshold", None), default_thr),
    )

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    predictor = _predictors[name]
    class_names = _class_names_per_model.get(name, CLASS_NAMES)
    outputs = predictor(img)
    instances = outputs.get("instances", None)
    if instances is not None:
        instances = instances.to("cpu")

    return _normalize_detections(instances, thr, class_names)


def dispose():
    """Release predictors (best effort)."""
    _predictors.clear()
    _model_paths.clear()
