"""
Contour/edge tools to extract retainer outer boundary and compute outward
arrow vectors for each detection.

The algorithm works on the full original image to avoid crop artifacts:
 - grayscale -> blur
 - method 'binary': Otsu threshold (+offset) then morphology close/open
 - choose external contour with the largest arc length
 - for each detection center, find nearest contour point and estimate tangent
   using neighbors; the arrow is the normal oriented away from the contour
   centroid (outward).
"""

from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2


DEFAULT_PARAMS: Dict[str, float] = {
    "method": "binary",       # binary only (binary or binary_inv)
    "blur": 21,               # odd kernel (will be coerced to odd)
    "thresh_offset": 12,     # add to Otsu threshold (can be negative)
    "morph": 1,               # morphology kernel size (pixels)
    "morph_iter": 1,          # morphology iterations
    "approx_eps": 1.0,        # polygon approximation epsilon (pixels)
    "arrow_len": 45.0,        # default arrow length in pixels
    "smooth_iters": 1,        # Chaikin smoothing iterations for contour
    "min_area": 0.0,          # ignore contours below this area (px^2); 0 to disable
}


def _to_gray(bgr):
    try:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        if bgr.ndim == 3 and bgr.shape[2] >= 3:
            return bgr[:, :, 2]
        return bgr


def extract_outer_contour(bgr, params: Optional[Dict] = None) -> Optional[np.ndarray]:
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    method = str(p.get("method", "binary") or "binary").lower()

    gray = _to_gray(bgr)
    k = int(max(1, int(p.get("blur", 5))))
    if k % 2 == 0:
        k += 1
    if k > 1:
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    morph = int(max(1, int(p.get("morph", 5))))
    mit = int(max(0, int(p.get("morph_iter", 1))))
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))

    def _cleanup(mask: np.ndarray) -> np.ndarray:
        clean = mask.copy()
        if mit > 0:
            clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, se, iterations=mit)
            clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, se, iterations=1)
        return clean

    def _largest_contour(mask: np.ndarray):
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        if not cnts:
            return None
        return max(cnts, key=lambda c: cv2.contourArea(c))

    def _score_contour(cnt, shape):
        if cnt is None:
            return -1.0
        area = cv2.contourArea(cnt)
        if area <= 0.0:
            return -1.0
        h, w = shape
        x, y, cw, ch = cv2.boundingRect(cnt)
        margin = min(x, y, w - (x + cw), h - (y + ch))
        border_penalty = 0.5 if margin < 5 else 1.0
        return area * border_penalty

    # Build binary masks (both inverted and non-inverted) because lighting can flip contrast.
    thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    off = float(p.get("thresh_offset", 0.0))
    level = max(0, min(255, int(thr + off)))
    _, mask_inv = cv2.threshold(gray, level, 255, cv2.THRESH_BINARY_INV)
    _, mask_pos = cv2.threshold(gray, level, 255, cv2.THRESH_BINARY)
    masks = [
        ("binary_inv", _cleanup(mask_inv)),
        ("binary", _cleanup(mask_pos)),
    ]

    # Force binary-only if requested (default)
    if method in {"binary", "binary_inv"}:
        masks = [m for m in masks if m[0] == method]
    elif method not in {"auto", "any", ""}:
        # Unknown method: fall back to auto over binary variants
        pass

    scored = []
    for label, mask in masks:
        cnt = _largest_contour(mask)
        if cnt is None:
            continue
        scored.append((label, cnt, _score_contour(cnt, gray.shape)))

    if not scored:
        return None

    scored.sort(key=lambda item: item[2], reverse=True)
    cnt = scored[0][1]

    # Optional smoothing via approxPolyDP
    eps = float(p.get("approx_eps", 2.0))
    if eps > 0:
        cnt = cv2.approxPolyDP(cnt, eps, closed=True)

    pts = cnt.reshape(-1, 2)

    # Optional Chaikin smoothing for a cleaner, less jagged boundary
    smooth_iters = int(max(0, int(p.get("smooth_iters", 0))))
    if smooth_iters > 0 and len(pts) >= 3:
        pts = _chaikin_smooth_closed(pts.astype(np.float32), smooth_iters)

    return pts


def _chaikin_smooth_closed(points: np.ndarray, iters: int) -> np.ndarray:
    """Chaikin corner cutting for closed polylines.

    points: Nx2 float32 array assumed closed (first and last implicitly connected).
    iters: number of smoothing iterations.
    Returns a new Nx2 float32 array.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return pts
    for _ in range(max(1, iters)):
        q = []
        n = len(pts)
        if n < 3:
            break
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            q.append(0.75 * p0 + 0.25 * p1)  # Q
            q.append(0.25 * p0 + 0.75 * p1)  # R
        pts = np.vstack(q).astype(np.float32)
    return pts


def compute_arrows_for_detections(
    bgr,
    detections: List[Dict],
    params: Optional[Dict] = None,
) -> Tuple[List[Dict], Optional[np.ndarray]]:
    """Return a list of arrow dicts matching detections and the contour used.

    Arrow dict has keys: 'anchor' (cx, cy), 'vec' (dx, dy).
    """
    h, w = bgr.shape[:2]
    cnt = extract_outer_contour(bgr, params)
    if cnt is None or len(cnt) < 3:
        # Fallback: outward along image centerline heuristic
        arrows = []
        for d in detections:
            b = d.get("bounds") if isinstance(d, dict) else None
            if not b:
                arrows.append({})
                continue
            x, y, bw, bh = b
            cx = float(x + bw / 2.0); cy = float(y + bh / 2.0)
            sign_x = 1.0 if cx >= w / 2.0 else -1.0
            vec = np.array([sign_x, 0.0], dtype=np.float32)
            L = float((params or {}).get("arrow_len", DEFAULT_PARAMS["arrow_len"]))
            arrows.append({"anchor": (cx, cy), "vec": (vec[0] * L, vec[1] * L)})
        return arrows, None

    # Build KD-like nearest search by simple argmin over contour points
    # Use contour centroid to consistently orient outward normals
    center_cnt = cnt.mean(axis=0).astype(np.float32)
    L = float((params or {}).get("arrow_len", DEFAULT_PARAMS["arrow_len"]))

    arrows = []
    for d in detections:
        b = d.get("bounds") if isinstance(d, dict) else None
        if not b:
            arrows.append({})
            continue
        x, y, bw, bh = b
        cx = float(x + bw / 2.0); cy = float(y + bh / 2.0)
        P = np.array([cx, cy], dtype=np.float32)
        # nearest contour index
        d2 = ((cnt - P) ** 2).sum(axis=1)
        idx = int(np.argmin(d2))
        # Estimate local tangent using immediate neighbors for stability
        i0 = (idx - 1) % len(cnt)
        i1 = (idx + 1) % len(cnt)
        t = (cnt[i1].astype(np.float32) - cnt[i0].astype(np.float32))
        if np.linalg.norm(t) < 1e-6:
            # fallback: wider baseline
            i0 = (idx - 3) % len(cnt)
            i1 = (idx + 3) % len(cnt)
            t = (cnt[i1].astype(np.float32) - cnt[i0].astype(np.float32))
        # Normal is perpendicular to tangent
        n = np.array([-t[1], t[0]], dtype=np.float32)
        nearest = cnt[idx].astype(np.float32)
        # Orient normal to point outward: away from contour centroid
        outward_dir = (nearest - center_cnt)
        if np.dot(n, outward_dir) < 0:
            n = -n
        nn = n / (np.linalg.norm(n) + 1e-6)
        vec = nn * L
        arrows.append({"anchor": (cx, cy), "vec": (float(vec[0]), float(vec[1]))})

    return arrows, cnt
