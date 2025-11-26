from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QLabel,
    QCheckBox,
    QHBoxLayout,
    QPushButton,
    QSplitter,
)


class ImagePreviewPanel(QWidget):
    overlay_toggled = pyqtSignal(bool)
    prev_requested = pyqtSignal()
    next_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter, stretch=1)

        # Left: Attachment Overview
        self.group_attachment = QGroupBox("3. Attachment Overview")
        left_v = QVBoxLayout(self.group_attachment)
        self.original_label = QLabel()
        self.original_label.setStyleSheet("background: black;")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(200, 150)
        self.original_label.setScaledContents(False)
        left_v.addWidget(self.original_label)
        splitter.addWidget(self.group_attachment)

        # Right: Front Inspection
        self.group_front = QGroupBox("4. Front Inspection")
        right_v = QVBoxLayout(self.group_front)

        header = QHBoxLayout()
        self.front_summary = QLabel("No inspection selected.")
        header.addWidget(self.front_summary, stretch=1)
        self.chk_overlay = QCheckBox("Show overlay")
        self.chk_overlay.setChecked(True)
        self.chk_overlay.toggled.connect(self.overlay_toggled)
        header.addWidget(self.chk_overlay)
        right_v.addLayout(header)

        self.front_label = QLabel()
        self.front_label.setStyleSheet("background: black;")
        self.front_label.setAlignment(Qt.AlignCenter)
        self.front_label.setMinimumSize(200, 150)
        self.front_label.setScaledContents(False)
        right_v.addWidget(self.front_label, stretch=1)

        nav = QHBoxLayout()
        self.bt_prev = QPushButton("Previous")
        self.bt_prev.clicked.connect(self.prev_requested)
        self.bt_next = QPushButton("Next")
        self.bt_next.clicked.connect(self.next_requested)
        nav.addWidget(self.bt_prev)
        nav.addStretch(1)
        nav.addWidget(self.bt_next)
        right_v.addLayout(nav)

        splitter.addWidget(self.group_front)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        # Keep base pixmaps (pre-overlay) for high-quality rescaling on resize
        self._original_base_pm = None
        self._front_base_pm = None
        self._front_overlay_enabled = True
        self._attachment_detections = []  # list of dicts: 'bounds','class','score', optional 'arrow'
        self._attachment_contour = None   # list of (x,y) points in original coords
        self._front_detections = []       # list of dicts for front view
        self._front_markers = []          # list of x positions (original pixel coords)
        self._draw_boxes = True

    # Public helpers
    def _apply_scaled_cover(self, label: QLabel, pm: QPixmap):
        if pm is None or pm.isNull():
            label.setText("Failed to load image.")
            label.setPixmap(QPixmap())
            return
        target_w = max(1, label.width())
        target_h = max(1, label.height())
        label.setPixmap(self._scale_and_crop(pm, target_w, target_h))

    def _scale_and_crop(self, pm: QPixmap, target_w: int, target_h: int) -> QPixmap:
        # First scale while preserving aspect ratio but ensuring we cover the target rect
        scaled = pm.scaled(target_w, target_h, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        # Then center-crop to exact label size to avoid letterboxing
        x = max(0, (scaled.width() - target_w) // 2)
        y = max(0, (scaled.height() - target_h) // 2)
        return scaled.copy(x, y, target_w, target_h)

    def _set_pixmap(self, label: QLabel, path: str):
        pm = QPixmap(path)
        self._apply_scaled_cover(label, pm)

    def _render_original(self):
        if self._original_base_pm is not None:
            # Scale and crop to fit
            target_w = max(1, self.original_label.width())
            target_h = max(1, self.original_label.height())
            base = self._original_base_pm
            composed = self._scale_and_crop(base, target_w, target_h)
            # Apply overlay if we have detections or a tuned contour
            if self._attachment_detections or (self._attachment_contour is not None):
                composed = self._apply_attachment_overlay(base, composed, target_w, target_h)
            self.original_label.setPixmap(composed)

    def _render_front(self):
        if self._front_base_pm is None:
            return
        # Scale and crop first to match the label size exactly, then draw overlay
        w = max(1, self.front_label.width())
        h = max(1, self.front_label.height())
        base = self._front_base_pm
        composed = self._scale_and_crop(base, w, h)
        if self._front_overlay_enabled:
            composed = self._apply_front_overlay(base, composed, w, h)
        self.front_label.setPixmap(composed)

    def _apply_front_overlay(self, base: QPixmap, composed: QPixmap, target_w: int, target_h: int) -> QPixmap:
        # Draw red center crosshair and optional detection boxes on the composed pixmap
        if composed is None or composed.isNull():
            return composed
        result = QPixmap(composed)
        painter = QPainter(result)
        painter.setRenderHint(QPainter.Antialiasing, True)
        # Compute mapping like in _apply_attachment_overlay
        bw, bh = base.width(), base.height()
        if bw <= 0 or bh <= 0:
            return result
        sx = target_w / bw
        sy = target_h / bh
        s = max(sx, sy)
        scaled_w = int(bw * s)
        scaled_h = int(bh * s)
        off_x = max(0, (scaled_w - target_w) // 2)
        off_y = max(0, (scaled_h - target_h) // 2)

        # Red guides
        w = result.width()
        h = result.height()
        pen = QPen(QColor(220, 0, 0))
        pen.setWidth(1)
        painter.setPen(pen)
        # Center vertical line
        painter.drawLine(int(w * 0.5), 0, int(w * 0.5), h)
        # Center horizontal line
        painter.drawLine(0, int(h * 0.5), w, int(h * 0.5))
        # Center circle
        r = int(min(w, h) * 0.06)
        cx = w // 2
        cy = h // 2
        painter.drawEllipse(cx - r, cy - r, 2 * r, 2 * r)
        # Optional detection rectangles for front view
        if self._front_detections:
            from PyQt5.QtGui import QFont
            box_pen = QPen(QColor(0, 200, 83))
            box_pen.setWidth(2)
            painter.setPen(box_pen)
            font = QFont(); font.setPointSize(9); painter.setFont(font)
            for d in self._front_detections:
                try:
                    b = d.get("bounds") or d.get("rect")
                    if not b:
                        continue
                    x, y, ww, hh = b
                    dx = int(x * s - off_x)
                    dy = int(y * s - off_y)
                    dw = int(ww * s)
                    dh = int(hh * s)
                    if dx + dw < 0 or dy + dh < 0 or dx > target_w or dy > target_h:
                        continue
                    painter.drawRect(dx, dy, dw, dh)
                    label = str(d.get("class", ""))
                    score = d.get("score")
                    if score is not None:
                        try:
                            label += f" {float(score):.2f}"
                        except Exception:
                            pass
                    if label:
                        metrics = painter.fontMetrics()
                        tw = metrics.width(label) + 6
                        th = metrics.height() + 4
                        painter.fillRect(dx, max(0, dy - th), tw, th, QColor(0, 200, 83, 180))
                        painter.setPen(QColor(255, 255, 255))
                        painter.drawText(dx + 3, dy - 4, label)
                        painter.setPen(box_pen)
                except Exception:
                    continue
        painter.end()
        # Draw blue markers (scaled x positions) on top of result
        if self._front_markers:
            painter = QPainter(result)
            painter.setRenderHint(QPainter.Antialiasing, True)
            # same mapping s/off_x/off_y computed above
            bw, bh = base.width(), base.height()
            sx = target_w / bw; sy = target_h / bh; s = max(sx, sy)
            scaled_w = int(bw * s); scaled_h = int(bh * s)
            off_x = max(0, (scaled_w - target_w) // 2)
            # place on horizontal midline
            dot_pen = QPen(QColor(0, 122, 204)); dot_pen.setWidth(2)
            painter.setPen(dot_pen)
            from PyQt5.QtGui import QBrush
            painter.setBrush(QColor(0, 122, 204))
            for x in self._front_markers:
                try:
                    dx = int(x * s - off_x)
                    dy = target_h // 2
                    painter.drawEllipse(dx - 5, dy - 5, 10, 10)
                except Exception:
                    continue
            painter.end()
        return result

    def _apply_attachment_overlay(self, base: QPixmap, composed: QPixmap, target_w: int, target_h: int) -> QPixmap:
        # Draw detection rectangles on the already scaled+cropped image.
        if composed is None or composed.isNull():
            return composed
        result = QPixmap(composed)
        from PyQt5.QtGui import QFont
        painter = QPainter(result)
        painter.setRenderHint(QPainter.Antialiasing, True)
        # Compute scale and crop offsets used by _scale_and_crop
        bw, bh = base.width(), base.height()
        if bw <= 0 or bh <= 0:
            return result
        sx = target_w / bw
        sy = target_h / bh
        s = max(sx, sy)
        scaled_w = int(bw * s)
        scaled_h = int(bh * s)
        off_x = max(0, (scaled_w - target_w) // 2)
        off_y = max(0, (scaled_h - target_h) // 2)
        # Styles
        pen = QPen(QColor(0, 200, 83))  # green
        pen.setWidth(2)
        painter.setPen(pen)
        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)
        # Draw each detection
        for d in self._attachment_detections:
            try:
                b = d.get("bounds") or d.get("rect")
                if not b:
                    continue
                x, y, w, h = b
                # Map to composed coordinates
                dx = int(x * s - off_x)
                dy = int(y * s - off_y)
                dw = int(w * s)
                dh = int(h * s)
                # Skip if fully outside
                if dx + dw < 0 or dy + dh < 0 or dx > target_w or dy > target_h:
                    continue
                # Optional polygon outline of detection
                poly = d.get("polygon")
                drew_poly = False
                if isinstance(poly, (list, tuple)) and len(poly) >= 3:
                    from PyQt5.QtGui import QPainterPath
                    path = QPainterPath()
                    for i, pt in enumerate(poly):
                        try:
                            px = int(pt[0] * s - off_x)
                            py = int(pt[1] * s - off_y)
                        except Exception:
                            continue
                        if i == 0:
                            path.moveTo(px, py)
                        else:
                            path.lineTo(px, py)
                    path.closeSubpath()
                    painter.drawPath(path)
                    drew_poly = True

                if self._draw_boxes and not drew_poly:
                    painter.drawRect(dx, dy, dw, dh)
                    label = str(d.get("class", ""))
                    score = d.get("score")
                    if score is not None:
                        try:
                            label += f" {float(score):.2f}"
                        except Exception:
                            pass
                    if label:
                        # Draw label with background box for readability
                        metrics = painter.fontMetrics()
                        tw = metrics.width(label) + 6
                        th = metrics.height() + 4
                        bg = QColor(0, 200, 83, 180)
                        painter.fillRect(dx, max(0, dy - th), tw, th, bg)
                        painter.setPen(QColor(255, 255, 255))
                        painter.drawText(dx + 3, dy - 4, label)
                        painter.setPen(pen)

                # Optional arrow overlay if provided
                arr = d.get("arrow")
                if isinstance(arr, dict):
                    anc = arr.get("anchor")
                    vec = arr.get("vec")
                    if anc and vec:
                        ax = int(anc[0] * s - off_x)
                        ay = int(anc[1] * s - off_y)
                        ex = int((anc[0] + vec[0]) * s - off_x)
                        ey = int((anc[1] + vec[1]) * s - off_y)
                        blue = QPen(QColor(0, 122, 204))
                        blue.setWidth(2)
                        painter.setPen(blue)
                        # Arrow line
                        painter.drawLine(ax, ay, ex, ey)
                        # Simple arrow head
                        import math
                        ang = math.atan2(ey - ay, ex - ax)
                        head_len = max(6, int(0.08 * (abs(ex - ax) + abs(ey - ay))))
                        a1 = ang + math.radians(155)
                        a2 = ang - math.radians(155)
                        hx1 = ex + int(head_len * math.cos(a1))
                        hy1 = ey + int(head_len * math.sin(a1))
                        hx2 = ex + int(head_len * math.cos(a2))
                        hy2 = ey + int(head_len * math.sin(a2))
                        painter.drawLine(ex, ey, hx1, hy1)
                        painter.drawLine(ex, ey, hx2, hy2)
                        painter.setPen(pen)
                        # Draw green index near the arrow base
                        idx_val = d.get("index") if isinstance(d, dict) else None
                        if idx_val is not None:
                            green = QPen(QColor(0, 200, 0))
                            painter.setPen(green)
                            painter.drawText(ax + 4, ay - 6, str(idx_val))
                            painter.setPen(pen)
                        # Draw yellow phi value near arrow end if present
                        phi = d.get("phi") if isinstance(d, dict) else None
                        if isinstance(phi, (int, float)):
                            ypen = QPen(QColor(255, 215, 0))
                        painter.setPen(ypen)
                        painter.drawText(ex + 6, ey, f"{phi:.3f}")
                        painter.setPen(pen)
            except Exception:
                continue
        # Optional tuned contour polyline
        try:
            cnt = self._attachment_contour
            if cnt is not None:
                blue = QPen(QColor(0, 122, 204))
                blue.setWidth(2)
                painter.setPen(blue)
                n = len(cnt)
                for i in range(max(0, n - 1)):
                    try:
                        px, py = cnt[i]
                        qx, qy = cnt[i + 1]
                        x1 = int(px * s - off_x); y1 = int(py * s - off_y)
                        x2 = int(qx * s - off_x); y2 = int(qy * s - off_y)
                        painter.drawLine(x1, y1, x2, y2)
                    except Exception:
                        pass
                if n >= 3:
                    try:
                        px, py = cnt[0]; qx, qy = cnt[-1]
                        x1 = int(px * s - off_x); y1 = int(py * s - off_y)
                        x2 = int(qx * s - off_x); y2 = int(qy * s - off_y)
                        painter.drawLine(x1, y1, x2, y2)
                    except Exception:
                        pass
        except Exception:
            pass
        painter.end()
        return result

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Rescale images on resize to keep zoom behavior similar to PictureBox Zoom
        self._render_original()
        self._render_front()

    def set_original_image(self, path: str):
        self._original_base_pm = QPixmap(path)
        self._render_original()

    def set_front_preview_image(self, path: str):
        self._front_base_pm = QPixmap(path)
        self._render_front()

    # New helpers for numpy images
    def set_original_np(self, pixmap: QPixmap):
        self._original_base_pm = pixmap
        self._render_original()

    def set_front_np(self, pixmap: QPixmap):
        self._front_base_pm = pixmap
        self._render_front()

    def set_overlay_enabled(self, enabled: bool):
        self._front_overlay_enabled = bool(enabled)
        self._render_front()

    def set_attachment_detections(self, detections):
        # Expect list of dicts with 'bounds': (x,y,w,h) and optional class/score
        self._attachment_detections = detections or []
        self._render_original()

    def set_draw_boxes(self, enabled: bool):
        self._draw_boxes = bool(enabled)
        self._render_original()

    def set_front_detections(self, detections):
        self._front_detections = detections or []
        self._render_front()

    def set_front_markers(self, xs):
        # xs: iterable of x positions in original front-image pixels
        try:
            self._front_markers = [int(x) for x in (xs or [])]
        except Exception:
            self._front_markers = []
        self._render_front()

    def set_attachment_contour(self, contour_points):
        # Accept list/ndarray of (x,y) points in original image coordinates
        try:
            self._attachment_contour = list(contour_points) if contour_points is not None else None
        except Exception:
            self._attachment_contour = None
        self._render_original()

    # Export helpers for saving composed attachment view (with overlays)
    def capture_attachment_view(self):
        if self._original_base_pm is None:
            return None
        target_w = max(1, self.original_label.width())
        target_h = max(1, self.original_label.height())
        base = self._original_base_pm
        composed = self._scale_and_crop(base, target_w, target_h)
        if self._attachment_detections or (self._attachment_contour is not None):
            composed = self._apply_attachment_overlay(base, composed, target_w, target_h)
        return composed

    def save_attachment_view(self, path: str) -> bool:
        pm = self.capture_attachment_view()
        if pm is None or pm.isNull():
            return False
        try:
            # Infer format from suffix; default to PNG
            suffix = (path.rsplit('.', 1)[-1] or 'png').upper() if '.' in path else 'PNG'
            return bool(pm.save(path, suffix))
        except Exception:
            return False

    def capture_front_view(self):
        if self._front_base_pm is None:
            return None
        target_w = max(1, self.front_label.width())
        target_h = max(1, self.front_label.height())
        base = self._front_base_pm
        composed = self._scale_and_crop(base, target_w, target_h)
        if self._front_overlay_enabled:
            composed = self._apply_front_overlay(base, composed, target_w, target_h)
        return composed

    def save_front_view(self, path: str) -> bool:
        pm = self.capture_front_view()
        if pm is None or pm.isNull():
            return False
        try:
            suffix = (path.rsplit('.', 1)[-1] or 'png').upper() if '.' in path else 'PNG'
            return bool(pm.save(path, suffix))
        except Exception:
            return False
