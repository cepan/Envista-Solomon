from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
)

from .camera_panel import CameraPanel
from .image_preview_panel import ImagePreviewPanel
from .turntable_panel import TurntablePanel
from services import camera_service, turntable_service
from services.config import settings, state, save_state


class InitWizard(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Initialization")
        self.setModal(True)
        self.resize(780, 540)

        root = QVBoxLayout(self)

        # Files group
        files = QGroupBox("Step 1 - Select Models/Files")
        vfiles = QVBoxLayout(files)

        def row(label: str):
            h = QHBoxLayout()
            h.addWidget(QLabel(label))
            le = QLineEdit()
            btn = QPushButton("Browse…")
            h.addWidget(le, stretch=1)
            h.addWidget(btn)
            return h, le, btn

        r1, self.le_attach, b1 = row("Attachment:")
        r2, self.le_front, b2 = row("Front Attachment:")
        r3, self.le_defect, b3 = row("Defect:")
        vfiles.addLayout(r1)
        vfiles.addLayout(r2)
        vfiles.addLayout(r3)

        def pick(le: QLineEdit):
            path, _ = QFileDialog.getOpenFileName(self, "Select file", "", "All Files (*.*)")
            if path:
                le.setText(path)

        b1.clicked.connect(lambda: pick(self.le_attach))
        b2.clicked.connect(lambda: pick(self.le_front))
        b3.clicked.connect(lambda: pick(self.le_defect))

        root.addWidget(files)

        # Cameras group
        self.cam_panel = CameraPanel()
        root.addWidget(self.cam_panel)

        # Live preview
        self.preview_panel = ImagePreviewPanel()
        root.addWidget(self.preview_panel)

        # Turntable group
        self.tt_panel = TurntablePanel()
        root.addWidget(self.tt_panel)

        # Bottom row
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        self.bt_begin = QPushButton("Begin Workflow")
        self.bt_begin.setEnabled(False)
        self.bt_cancel = QPushButton("Cancel")
        bottom.addWidget(self.bt_cancel)
        bottom.addWidget(self.bt_begin)
        root.addLayout(bottom)

        # Wire cameras
        self.cam_panel.refresh_requested.connect(self.on_cam_refresh)
        self.cam_panel.connect_requested.connect(self.on_cam_connect)
        self.cam_panel.disconnect_requested.connect(self.on_cam_disconnect)
        self.cam_panel.capture_requested.connect(self.on_cam_capture)
        self.on_cam_refresh()

        # Wire turntable
        self.tt_panel.refresh_requested.connect(self.on_tt_refresh)
        self.tt_panel.connect_requested.connect(self.on_tt_connect)
        self.tt_panel.disconnect_requested.connect(self.on_tt_disconnect)
        self.tt_panel.home_requested.connect(self.on_tt_home)
        self.tt_panel.rotate_requested.connect(self.on_tt_rotate)
        self.on_tt_refresh()

        # Buttons
        self.bt_cancel.clicked.connect(self.reject)
        self.bt_begin.clicked.connect(self.accept)

        # Pre-fill from persisted state
        st = state()
        if st.attachment_path:
            self.le_attach.setText(st.attachment_path)
        if st.front_attachment_path:
            self.le_front.setText(st.front_attachment_path)
        if st.defect_path:
            self.le_defect.setText(st.defect_path)

        # Validation timerless check
        self._update_ready()
        self.le_attach.textChanged.connect(self._update_ready)
        self.le_front.textChanged.connect(self._update_ready)
        self.le_defect.textChanged.connect(self._update_ready)

    # Cameras
    def on_cam_refresh(self):
        try:
            devices = camera_service.enumerate_devices()
            self.cam_panel.set_devices(devices)
            # Restore last selections (no auto-connect)
            st = state()
            if st.camera_top_index is not None:
                self.cam_panel.set_selected_index("Top", int(st.camera_top_index))
            if st.camera_front_index is not None:
                self.cam_panel.set_selected_index("Front", int(st.camera_front_index))
        except Exception:
            pass

    def on_cam_connect(self, role: str, index: int):
        # prevent same device for both roles
        other = "Front" if role == "Top" else "Top"
        if camera_service.get_connected_index(other) == index:
            return
        if camera_service.connect(role, index):
            name = ""
            self.cam_panel.set_connected(role, True, name)
            # Persist selection
            st = state()
            if role == "Top":
                st.camera_top_index = index
            else:
                st.camera_front_index = index
            save_state()
            # Auto-capture on connect
            self.on_cam_capture(role)
        self._update_ready()

    def on_cam_disconnect(self, role: str):
        camera_service.disconnect(role)
        self.cam_panel.set_connected(role, False)
        self._update_ready()

    def on_cam_capture(self, role: str):
        try:
            from services import camera_manager as _cammgr
            frame = _cammgr.capture(role)
            from .qt_image import np_bgr_to_qpixmap
            pm = np_bgr_to_qpixmap(frame)
            if role == "Top":
                self.preview_panel.set_original_np(pm)
                settings().top_preview_np = frame
            else:
                self.preview_panel.set_front_np(pm)
                settings().front_preview_np = frame
        except Exception:
            pass

    # Turntable
    def on_tt_refresh(self):
        try:
            ports = turntable_service.refresh_devices()
            self.tt_panel.set_ports(ports)
            st = state()
            if st.turntable_port:
                idx = self.tt_panel.port_combo.findText(st.turntable_port)
                if idx >= 0:
                    self.tt_panel.port_combo.setCurrentIndex(idx)
        except Exception:
            pass

    def on_tt_connect(self, port: str):
        if turntable_service.connect(port):
            self.tt_panel.set_connected(True, port)
            st = state()
            st.turntable_port = port
            st.turntable_step = float(self.tt_panel.step.value())
            save_state()
        self._update_ready()

    def on_tt_disconnect(self):
        turntable_service.disconnect()
        self.tt_panel.set_connected(False)
        self._update_ready()

    def on_tt_home(self):
        import threading
        def run():
            res = turntable_service.home()
            self.tt_panel.set_status(res.message)
            self._update_ready()
        threading.Thread(target=run, daemon=True).start()

    def on_tt_rotate(self, angle: float):
        import threading
        def run():
            try:
                msg = turntable_service.move_relative(angle)
                self.tt_panel.set_status(msg)
            except Exception as ex:
                self.tt_panel.set_status(str(ex))
        threading.Thread(target=run, daemon=True).start()

    def _update_ready(self):
        # Persist into settings
        s = settings()
        s.attachment_path = self.le_attach.text().strip() or None
        s.front_attachment_path = self.le_front.text().strip() or None
        s.defect_path = self.le_defect.text().strip() or None
        st = state()
        st.attachment_path = s.attachment_path
        st.front_attachment_path = s.front_attachment_path
        st.defect_path = s.defect_path
        save_state()

        cams_ok = (
            camera_service.get_connected_index("Top") is not None and
            camera_service.get_connected_index("Front") is not None
        )
        tt_ok = turntable_service.is_connected()
        # Files optional for now
        self.bt_begin.setEnabled(cams_ok and tt_ok)

    def accept(self):
        # Save final selections before closing
        st = state()
        st.attachment_path = self.le_attach.text().strip() or None
        st.front_attachment_path = self.le_front.text().strip() or None
        st.defect_path = self.le_defect.text().strip() or None
        st.turntable_step = float(self.tt_panel.step.value())
        save_state()
        super().accept()
