#!/usr/bin/env python3
"""
live_blob_tool.py
-----------------
A zero-install (besides OpenCV) Python tool for **live** image thresholding and blob detection.
- Works with a webcam (default) or a single image file (static preview).
- Interactive trackbars let you tune thresholding and blob-detector parameters in real time.
- Saves annotated frames with the 's' key; quit with 'q' or ESC.

Usage:
  python live_blob_tool.py              # use default webcam (index 0)
  python live_blob_tool.py --source 1   # use a different camera index
  python live_blob_tool.py --image path/to/image.jpg   # process a static image interactively
  python live_blob_tool.py --video path/to/video.mp4   # (optional) process a video file

Dependencies:
  pip install opencv-python numpy

Notes:
  - Blob detection uses OpenCV's SimpleBlobDetector on the (optionally) thresholded image,
    so your threshold + morphology choices directly influence which blobs are found.
  - If your blobs are dark on light, set "Blob color" to 'Dark' (1). For light blobs, set to 'Light' (0).
"""

import argparse
import sys
import os
import cv2
import numpy as np
from typing import Tuple

WIN_PREVIEW = "Live Blob Detection"
WIN_THRESH = "Threshold"
WIN_CTL = "Controls"

def _ensure_window(name: str) -> None:
    try:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    except cv2.error:
        # Some environments need destroyAllWindows first
        cv2.destroyAllWindows()
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)

def _odd_from_slider(v: int, minimum: int = 3) -> int:
    """Map slider int -> odd number >= minimum."""
    if v < minimum:
        v = minimum
    if v % 2 == 0:
        v += 1
    return v

def create_trackbars() -> None:
    _ensure_window(WIN_CTL)
    # Threshold section
    cv2.createTrackbar("Thresh type (0..4)", WIN_CTL, 0, 4, lambda v: None)
    # 0=Binary, 1=BinaryInv, 2=AdaptiveMean, 3=AdaptiveGaussian, 4=Otsu
    cv2.createTrackbar("Manual thresh", WIN_CTL, 127, 255, lambda v: None)
    cv2.createTrackbar("Adaptive block size", WIN_CTL, 11, 99, lambda v: None)  # odd only internally
    cv2.createTrackbar("Adaptive C", WIN_CTL, 2, 20, lambda v: None)
    cv2.createTrackbar("Gaussian blur (px)", WIN_CTL, 1, 25, lambda v: None)
    cv2.createTrackbar("Open iters", WIN_CTL, 0, 10, lambda v: None)
    cv2.createTrackbar("Close iters", WIN_CTL, 0, 10, lambda v: None)

    # Blob detector section
    cv2.createTrackbar("Blob color (0=Light,1=Dark)", WIN_CTL, 1, 1, lambda v: None)
    cv2.createTrackbar("minArea", WIN_CTL, 50, 100000, lambda v: None)
    cv2.createTrackbar("maxArea", WIN_CTL, 5000, 200000, lambda v: None)
    cv2.createTrackbar("minCircularity x100", WIN_CTL, 0, 100, lambda v: None)
    cv2.createTrackbar("minInertia x100", WIN_CTL, 0, 100, lambda v: None)
    cv2.createTrackbar("minConvexity x100", WIN_CTL, 0, 100, lambda v: None)
    cv2.createTrackbar("minThreshold", WIN_CTL, 10, 255, lambda v: None)
    cv2.createTrackbar("maxThreshold", WIN_CTL, 200, 255, lambda v: None)
    cv2.createTrackbar("thresholdStep", WIN_CTL, 10, 50, lambda v: None)

def read_trackbar_params() -> dict:
    params = {}
    tt = cv2.getTrackbarPos("Thresh type (0..4)", WIN_CTL)
    params["thresh_type"] = int(np.clip(tt, 0, 4))
    params["thresh"] = cv2.getTrackbarPos("Manual thresh", WIN_CTL)
    params["block_size"] = _odd_from_slider(cv2.getTrackbarPos("Adaptive block size", WIN_CTL), 3)
    params["C"] = cv2.getTrackbarPos("Adaptive C", WIN_CTL)
    params["blur_px"] = cv2.getTrackbarPos("Gaussian blur (px)", WIN_CTL)
    params["open_iter"] = cv2.getTrackbarPos("Open iters", WIN_CTL)
    params["close_iter"] = cv2.getTrackbarPos("Close iters", WIN_CTL)

    params["blob_dark"] = cv2.getTrackbarPos("Blob color (0=Light,1=Dark)", WIN_CTL) == 1
    params["minArea"] = max(1, cv2.getTrackbarPos("minArea", WIN_CTL))
    params["maxArea"] = max(params["minArea"]+1, cv2.getTrackbarPos("maxArea", WIN_CTL))
    params["minCircularity"] = cv2.getTrackbarPos("minCircularity x100", WIN_CTL) / 100.0
    params["minInertia"] = cv2.getTrackbarPos("minInertia x100", WIN_CTL) / 100.0
    params["minConvexity"] = cv2.getTrackbarPos("minConvexity x100", WIN_CTL) / 100.0
    params["sbd_minThreshold"] = cv2.getTrackbarPos("minThreshold", WIN_CTL)
    params["sbd_maxThreshold"] = max(params["sbd_minThreshold"]+1, cv2.getTrackbarPos("maxThreshold", WIN_CTL))
    params["sbd_thresholdStep"] = max(1, cv2.getTrackbarPos("thresholdStep", WIN_CTL))
    return params

def apply_threshold(gray: np.ndarray, p: dict) -> np.ndarray:
    g = gray.copy()
    # Optional blur to reduce noise
    k = max(0, int(p["blur_px"]))
    if k >= 2:
        k = _odd_from_slider(k if k % 2 else k-1, 1)
        g = cv2.GaussianBlur(g, (k, k), 0)

    ttype = p["thresh_type"]
    if ttype == 0:  # Binary
        _, th = cv2.threshold(g, p["thresh"], 255, cv2.THRESH_BINARY)
    elif ttype == 1:  # Binary Inv
        _, th = cv2.threshold(g, p["thresh"], 255, cv2.THRESH_BINARY_INV)
    elif ttype in (2, 3):  # Adaptive
        method = cv2.ADAPTIVE_THRESH_MEAN_C if ttype == 2 else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        bs = max(3, p["block_size"] | 1)  # ensure odd
        th = cv2.adaptiveThreshold(g, 255, method, cv2.THRESH_BINARY, bs, p["C"])
    else:  # Otsu
        _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphology
    if p["open_iter"] > 0:
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=p["open_iter"])
    if p["close_iter"] > 0:
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=p["close_iter"])
    return th

def build_blob_detector(p: dict) -> cv2.SimpleBlobDetector:
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = float(p["sbd_minThreshold"])
    params.maxThreshold = float(p["sbd_maxThreshold"])
    params.thresholdStep = float(p["sbd_thresholdStep"])

    params.filterByArea = True
    params.minArea = float(p["minArea"])
    params.maxArea = float(p["maxArea"])

    params.filterByCircularity = p["minCircularity"] > 0.0
    params.minCircularity = float(p["minCircularity"])

    params.filterByInertia = p["minInertia"] > 0.0
    params.minInertiaRatio = float(p["minInertia"])

    params.filterByConvexity = p["minConvexity"] > 0.0
    params.minConvexity = float(p["minConvexity"])

    params.filterByColor = True
    params.blobColor = 0 if p["blob_dark"] else 255

    try:
        detector = cv2.SimpleBlobDetector_create(params)
    except AttributeError:
        # Older OpenCV versions
        detector = cv2.SimpleBlobDetector(params)
    return detector

def annotate(frame_bgr: np.ndarray, keypoints: list, fps: float, p: dict) -> np.ndarray:
    out = frame_bgr.copy()
    out = cv2.drawKeypoints(out, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    n = len(keypoints)
    h, w = out.shape[:2]
    legend = [
        f"Blobs: {n} | FPS: {fps:.1f}",
        f"Area: [{int(p['minArea'])}, {int(p['maxArea'])}]  Circ>={p['minCircularity']:.2f}  Iner>={p['minInertia']:.2f}  Conv>={p['minConvexity']:.2f}",
        f"SBD thresholds: {int(p['sbd_minThreshold'])}-{int(p['sbd_maxThreshold'])} step {int(p['sbd_thresholdStep'])}  Color: {'Dark' if p['blob_dark'] else 'Light'}"
    ]
    y = 24
    for line in legend:
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        y += 24
    return out

def open_source(args) -> Tuple[cv2.VideoCapture, bool]:
    if args.image:
        img = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[ERROR] Could not load image: {args.image}", file=sys.stderr)
            sys.exit(1)
        return img, False  # static image mode
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {args.video}", file=sys.stderr)
            sys.exit(1)
        return cap, True
    # webcam
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {args.source}", file=sys.stderr)
        sys.exit(1)
    return cap, True

def main():
    parser = argparse.ArgumentParser(description="Live thresholding + blob detection with OpenCV.")
    parser.add_argument("--source", type=int, default=0, help="Webcam index (default: 0).")
    parser.add_argument("--image", type=str, help="Path to a single image for static interactive tuning.")
    parser.add_argument("--video", type=str, help="Path to a video file.")
    parser.add_argument("--max-width", type=int, default=1280, help="Resize display if wider than this (0 to disable).")
    args = parser.parse_args()

    src, streaming = open_source(args)

    _ensure_window(WIN_PREVIEW)
    _ensure_window(WIN_THRESH)
    create_trackbars()

    prev_time = cv2.getTickCount()
    tick_freq = cv2.getTickFrequency()

    while True:
        if streaming:
            ok, frame = src.read()
            if not ok:
                print("[INFO] End of stream.")
                break
        else:
            frame = src.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p = read_trackbar_params()
        th = apply_threshold(gray, p)

        # Build detector & detect on the thresholded image for direct control
        detector = build_blob_detector(p)
        keypoints = detector.detect(th)

        # Prepare display
        fps = 0.0
        now = cv2.getTickCount()
        dt = (now - prev_time) / tick_freq
        if dt > 0:
            fps = 1.0 / dt
        prev_time = now

        annotated = annotate(frame, keypoints, fps, p)

        # Optional resize for display comfort
        if args.max_width and annotated.shape[1] > args.max_width:
            scale = args.max_width / annotated.shape[1]
            annotated = cv2.resize(annotated, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            th_disp = cv2.resize(th, (annotated.shape[1], annotated.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            th_disp = th

        cv2.imshow(WIN_PREVIEW, annotated)
        cv2.imshow(WIN_THRESH, th_disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC or 'q'
            break
        elif key == ord('s'):
            # Save annotated frame and threshold image
            os.makedirs("captures", exist_ok=True)
            base = "captures/capture"
            i = 1
            while os.path.exists(f"{base}_{i:03d}.png"):
                i += 1
            out_path = f"{base}_{i:03d}.png"
            th_path = f"{base}_{i:03d}_th.png"
            cv2.imwrite(out_path, annotated)
            cv2.imwrite(th_path, th_disp)
            print(f"[SAVED] {out_path} and {th_path}")
        # In static image mode, loop without reading a new frame
        if not streaming:
            # Small delay to keep UI responsive
            cv2.waitKey(10)

    if streaming:
        try:
            src.release()
        except Exception:
            pass
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
