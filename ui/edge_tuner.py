from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout, QFileDialog,
    QMessageBox,
)

import cv2
import numpy as np

from services import contour_tools as ct
from services import camera_manager as cammgr


class EdgeTunerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edge/Contour Tuner")
        self.resize(900, 680)
        self._img_path = None
        self._img_np = None
        self._last_vis_np = None
        self._last_cnt = None
        self._last_src_np = None

        # Controls
        frm = QFormLayout()
        self.combo_method = QComboBox()
        self.combo_method.addItems(["binary"])  # fixed to binary
        self.combo_method.setCurrentIndex(0)
        self.combo_method.setEnabled(False)
        frm.addRow("Method", self.combo_method)

        self.spin_blur = QSpinBox(); self.spin_blur.setRange(1, 31); self.spin_blur.setSingleStep(2); self.spin_blur.setValue(int(ct.DEFAULT_PARAMS.get("blur", 5)))
        frm.addRow("Blur (odd)", self.spin_blur)

        self.spin_morph = QSpinBox(); self.spin_morph.setRange(1, 31); self.spin_morph.setValue(int(ct.DEFAULT_PARAMS.get("morph", 3)))
        frm.addRow("Morph kernel", self.spin_morph)

        self.spin_morph_iter = QSpinBox(); self.spin_morph_iter.setRange(0, 5); self.spin_morph_iter.setValue(int(ct.DEFAULT_PARAMS.get("morph_iter", 1)))
        frm.addRow("Morph iter", self.spin_morph_iter)

        self.spin_eps = QDoubleSpinBox(); self.spin_eps.setRange(0.0, 10.0); self.spin_eps.setDecimals(1); self.spin_eps.setValue(float(ct.DEFAULT_PARAMS.get("approx_eps", 2.0)))
        frm.addRow("Approx eps", self.spin_eps)

        self.spin_smooth = QSpinBox(); self.spin_smooth.setRange(0, 10); self.spin_smooth.setValue(int(ct.DEFAULT_PARAMS.get("smooth_iters", 1)))
        frm.addRow("Smooth iter", self.spin_smooth)

        self.spin_th_off = QDoubleSpinBox(); self.spin_th_off.setRange(-50.0, 50.0); self.spin_th_off.setDecimals(1); self.spin_th_off.setValue(float(ct.DEFAULT_PARAMS.get("thresh_offset", 0.0)))
        frm.addRow("Thresh offset", self.spin_th_off)

        self.spin_arrow = QDoubleSpinBox(); self.spin_arrow.setRange(5.0, 200.0); self.spin_arrow.setValue(float(ct.DEFAULT_PARAMS.get("arrow_len", 60.0)))
        frm.addRow("Arrow length", self.spin_arrow)

        box = QGroupBox("Parameters"); box.setLayout(frm)

        # Widgets
        self._path_label = QLabel("Image: (none)")
        self._path_label.setStyleSheet("color: #888;")
        self._preview = QLabel("Choose an image or Capture Top")
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setStyleSheet("background: #111;")

        # Buttons
        self.bt_open = QPushButton("Open Image")
        self.bt_preview = QPushButton("Preview Contour")
        self.bt_apply = QPushButton("Apply To Overlay")
        self.bt_ok = QPushButton("Close")
        self.bt_capture = QPushButton("Capture Top")

        self.bt_ok.clicked.connect(self.accept)
        self.bt_open.clicked.connect(self._choose_image)
        self.bt_preview.clicked.connect(self._preview_contour_sync)
        self.bt_apply.clicked.connect(self._apply_overlay)
        self.bt_capture.clicked.connect(self._capture_top)

        layout = QVBoxLayout(self)
        layout.addWidget(box)
        layout.addWidget(self._path_label)
        layout.addWidget(self._preview, 1)
        layout.addLayout(self._hbox(self.bt_open, self.bt_preview, self.bt_apply, self.bt_ok))
        layout.addLayout(self._hbox(self.bt_capture))

        # Debounce preview
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(200)
        self._debounce.timeout.connect(self._preview_contour_sync)
        for w in (self.spin_blur, self.spin_morph, self.spin_morph_iter, self.spin_eps, self.spin_smooth, self.spin_th_off, self.spin_arrow):
            w.valueChanged.connect(lambda _=None: self._on_param_changed())

    def _hbox(self, *widgets):
        hb = QHBoxLayout(); hb.setSpacing(8)
        for w in widgets: hb.addWidget(w)
        hb.addStretch(1)
        return hb

    # --- Public setters expected by callers (MainWindow) ---
    def set_image_path(self, path: str) -> bool:
        """Load an image from disk and set it as the tuner source.
        Returns True on success.
        """
        if not path:
            return False
        try:
            img = cv2.imread(path)
            if img is None:
                return False
            self._img_np = img
            self._img_path = path
            self._path_label.setText(f"Image: {path}")
            self._set_preview(img)
            return True
        except Exception:
            return False

    def set_image_np(self, bgr) -> bool:
        """Set a numpy BGR image directly as the tuner source.
        Returns True on success.
        """
        try:
            if bgr is None:
                return False
            self._img_np = bgr.copy()
            self._img_path = None
            self._path_label.setText("Image: Top camera (live)")
            self._set_preview(self._img_np)
            return True
        except Exception:
            return False

    def _on_param_changed(self):
        try:
            from services.config import state as _state, save_state as _save
            st = _state(); st.contour_params = dict(self.params()); _save()
        except Exception:
            pass
        self._debounce.start()

    def _choose_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose Image", "", "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*.*)")
        if not path:
            return
        try:
            img = cv2.imread(path)
            if img is None:
                raise RuntimeError("Failed to read image")
            self._img_np = img
            self._img_path = path
            self._path_label.setText(f"Image: {path}")
            self._set_preview(img)
        except Exception as ex:
            QMessageBox.warning(self, "Open Image", f"Failed to open image.\n{ex}")

    def _get_best_image(self):
        if self._img_np is not None:
            return self._img_np
        if self._img_path:
            try:
                return cv2.imread(self._img_path)
            except Exception:
                return None
        return None

    def _capture_top(self):
        try:
            img = cammgr.capture("Top") if cammgr.is_connected("Top") else None
            if img is None:
                self._preview.setText("Top camera not available.")
                return
            self._img_np = img
            self._img_path = None
            self._path_label.setText("Image: Top camera (live)")
            self._set_preview(img)
            self._preview_contour_sync()
        except Exception as ex:
            self._preview.setText(f"Capture failed: {ex}")

    def _preview_contour(self):
        self._preview_contour_sync()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._preview_contour_sync)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        try:
            if self._last_vis_np is not None:
                self._set_preview(self._last_vis_np)
            elif self._img_np is not None:
                self._set_preview(self._img_np)
        except Exception:
            pass

    def _preview_contour_sync(self):
        try:
            img = self._get_best_image()
            if img is None:
                self._preview.setText("No image available. Connect Top camera or open an image.")
                return
            self._img_np = img
            cnt = ct.extract_outer_contour(img, self.params())
            self._last_cnt = cnt
            self._last_src_np = img
            vis = self._render_contour_preview(img, cnt)
            self._last_vis_np = vis
            self._set_preview(vis)
        except Exception as ex:
            self._preview.setText(f"Preview failed: {ex}")

    def params(self) -> dict:
        return {
            "method": self.combo_method.currentText(),
            "blur": int(self.spin_blur.value()),
            "morph": int(self.spin_morph.value()),
            "morph_iter": int(self.spin_morph_iter.value()),
            "approx_eps": float(self.spin_eps.value()),
            "smooth_iters": int(self.spin_smooth.value()),
            "thresh_offset": float(self.spin_th_off.value()),
            "arrow_len": float(self.spin_arrow.value()),
        }

    def _apply_overlay(self):
        # Persist and try to push overlay to main preview
        try:
            from services.config import state as _state, save_state as _save
            st = _state(); st.contour_params = dict(self.params()); _save()
        except Exception:
            pass
        try:
            parent = self.parent()
            if parent is not None:
                src = self._get_best_image()
                if src is not None:
                    try:
                        from ui.qt_image import np_bgr_to_qpixmap
                        parent.preview_panel.set_original_np(np_bgr_to_qpixmap(src))
                    except Exception:
                        pass
                    cnt = ct.extract_outer_contour(src, self.params())
                    try:
                        parent.preview_panel.set_attachment_contour(cnt if cnt is not None else [])
                        parent.workflow_tab.append_log("[Tuner] Applied contour overlay from tuner to main preview.")
                    except Exception:
                        pass
        except Exception:
            pass
        self.accept()

    def _set_preview(self, bgr):
        try:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qi = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
            pm = QPixmap.fromImage(qi)
            wlbl = max(1, self._preview.width())
            hlbl = max(1, self._preview.height())
            self._preview.setPixmap(pm.scaled(wlbl, hlbl, Qt.KeepAspectRatio, transformMode=Qt.FastTransformation))
        except Exception as ex:
            self._preview.setText(f"Preview failed: {ex}")

    def _render_contour_preview(self, img, cnt):
        vis = img.copy()
        if cnt is None or len(cnt) < 2:
            return vis
        pts = cnt.reshape(-1, 1, 2).astype(np.int32)
        h, w = img.shape[:2]
        thickness = max(2, int(min(h, w) * 0.004))
        color = (204, 122, 0)  # BGR
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        return vis
