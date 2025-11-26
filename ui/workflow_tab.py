from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QPlainTextEdit,
    QScrollArea,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QSpinBox,
    QLineEdit,
)
from PyQt5.QtCore import Qt, pyqtSignal
from services.config import state as _state, save_state as _save_state
from .camera_panel import CameraPanel
from .turntable_panel import TurntablePanel
from .linear_axis_panel import LinearAxisPanel


class WorkflowTab(QWidget):
    load_image_requested = pyqtSignal()
    run_detection_requested = pyqtSignal()
    open_tuner_requested = pyqtSignal()
    load_attachment_requested = pyqtSignal(str)
    load_front_requested = pyqtSignal(str)
    load_defect_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Scrollable container to match WinForms AutoScroll panel
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        scroll.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # Step 1 - Selected Models
        self.group_files = QGroupBox("Step 1 - Selected Models")
        files_layout = QVBoxLayout(self.group_files)

        def file_row(label_text: str):
            row = QHBoxLayout()
            label = QLabel(label_text)
            edit = QTextEdit()
            edit.setFixedHeight(30)
            edit.setAcceptRichText(False)
            btn = QPushButton("Load")
            self._apply_primary_style(btn, active=False)
            row.addWidget(label)
            row.addWidget(edit, 1)
            row.addWidget(btn)
            return row, edit, btn

        rA, self.edit_attach, self.bt_attach_load = file_row("Attachment:")
        rF, self.edit_front, self.bt_front_load = file_row("Front Attachment:")
        rD, self.edit_defect, self.bt_defect_load = file_row("Defect:")
        files_layout.addLayout(rA)
        files_layout.addLayout(rF)
        files_layout.addLayout(rD)
        layout.addWidget(self.group_files)

        # Wire load buttons to main window
        self.bt_attach_load.clicked.connect(lambda: self.load_attachment_requested.emit(self.edit_attach.toPlainText().strip()))
        self.bt_front_load.clicked.connect(lambda: self.load_front_requested.emit(self.edit_front.toPlainText().strip()))
        self.bt_defect_load.clicked.connect(lambda: self.load_defect_requested.emit(self.edit_defect.toPlainText().strip()))

        # Optional: Upload Image section (hidden by default)
        self.group_step2 = QGroupBox("Step 2 - Upload Image")
        self.group_step2.setVisible(False)
        step2_layout = QVBoxLayout(self.group_step2)
        self.bt_load_img = QPushButton("Choose Image…")
        self.bt_load_img.clicked.connect(self.load_image_requested.emit)
        step2_layout.addWidget(self.bt_load_img)
        layout.addWidget(self.group_step2)

        # Cameras
        self.camera_panel = CameraPanel()
        layout.addWidget(self.camera_panel)

        # Turntable
        self.turntable_panel = TurntablePanel()
        layout.addWidget(self.turntable_panel)

        # Linear axis (front camera)
        self.linear_axis_panel = LinearAxisPanel()
        layout.addWidget(self.linear_axis_panel)

        # Run Detection
        self.group_detect = QGroupBox("Run Detection")
        detect_layout = QVBoxLayout(self.group_detect)
        # Part ID row
        pid_row = QHBoxLayout()
        pid_label = QLabel("Part ID:")
        self.edit_part_id = QLineEdit()
        try:
            self.edit_part_id.setText(str(getattr(_state(), 'part_id', '') or ''))
        except Exception:
            pass
        pid_row.addWidget(pid_label)
        pid_row.addWidget(self.edit_part_id, 1)
        detect_layout.addLayout(pid_row)
        self.bt_detect = QPushButton("Run Detection")
        self.bt_detect.clicked.connect(self.run_detection_requested.emit)
        detect_layout.addWidget(self.bt_detect)
        # Optional: open edge/contour tuner dialog
        self.bt_tuner = QPushButton("Tune Edge/Contour…")
        self.bt_tuner.clicked.connect(self.open_tuner_requested.emit)
        detect_layout.addWidget(self.bt_tuner)
        layout.addWidget(self.group_detect)

        # Step 2 crop settings
        self.group_crop = QGroupBox("Step 2 - Crop Settings")
        crop_layout = QHBoxLayout(self.group_crop)
        crop_label = QLabel("Square crop size (px):")
        self.crop_size = QSpinBox()
        self.crop_size.setRange(100, 8192)
        try:
            st0 = _state()
            default_cs = int(getattr(st0, 'step2_crop_size', None) or 1600)
        except Exception:
            default_cs = 1600
        self.crop_size.setValue(default_cs)
        self.crop_size.setSingleStep(50)
        crop_layout.addWidget(crop_label)
        crop_layout.addWidget(self.crop_size, 1)
        layout.addWidget(self.group_crop)
        # Persist crop size on change
        def _on_crop_changed(v):
            try:
                st = _state(); st.step2_crop_size = int(v); _save_state()
                self.append_log(f"[Step2] Crop size set to {int(v)} px")
            except Exception:
                pass
        self.crop_size.valueChanged.connect(_on_crop_changed)

        # Light controller settings
        self.group_light = QGroupBox("Light Controller")
        light_layout = QHBoxLayout(self.group_light)
        light_layout.addWidget(QLabel("IP:"))
        self.edit_light_ip = QTextEdit(); self.edit_light_ip.setFixedHeight(28); self.edit_light_ip.setAcceptRichText(False)
        try:
            self.edit_light_ip.setPlainText(str(getattr(_state(), 'light_ip', '') or ''))
        except Exception:
            pass
        light_layout.addWidget(self.edit_light_ip, 1)
        light_layout.addWidget(QLabel("Top (mA):"))
        self.spin_top_ma = QSpinBox(); self.spin_top_ma.setRange(0, 250); self.spin_top_ma.setSingleStep(5)
        try:
            self.spin_top_ma.setValue(int(getattr(_state(), 'top_current_ma', 0) or 0))
        except Exception:
            self.spin_top_ma.setValue(0)
        light_layout.addWidget(self.spin_top_ma)
        light_layout.addWidget(QLabel("Front (mA):"))
        self.spin_front_ma = QSpinBox(); self.spin_front_ma.setRange(0, 250); self.spin_front_ma.setSingleStep(5)
        try:
            self.spin_front_ma.setValue(int(getattr(_state(), 'front_current_ma', 0) or 0))
        except Exception:
            self.spin_front_ma.setValue(0)
        light_layout.addWidget(self.spin_front_ma)
        # Dwell (ms) setting; when brightness changes, capture waits this long
        light_layout.addWidget(QLabel("Dwell (ms):"))
        self.spin_dwell = QSpinBox(); self.spin_dwell.setRange(0, 2000); self.spin_dwell.setSingleStep(10)
        try:
            self.spin_dwell.setValue(int(getattr(_state(), 'light_dwell_ms', 60) or 60))
        except Exception:
            self.spin_dwell.setValue(60)
        light_layout.addWidget(self.spin_dwell)
        layout.addWidget(self.group_light)

        def _persist_light():
            try:
                st = _state()
                st.light_ip = self.edit_light_ip.toPlainText().strip()
                st.top_current_ma = int(self.spin_top_ma.value())
                st.front_current_ma = int(self.spin_front_ma.value())
                st.light_dwell_ms = int(self.spin_dwell.value())
                _save_state()
                dw = int(st.light_dwell_ms or 0)
                self.append_log(f"[Light] ip={st.light_ip} top={st.top_current_ma}mA front={st.front_current_ma}mA dwell={dw}ms")
            except Exception:
                pass
        self.edit_light_ip.textChanged.connect(lambda: _persist_light())
        self.spin_top_ma.valueChanged.connect(lambda _v: _persist_light())
        self.spin_front_ma.valueChanged.connect(lambda _v: _persist_light())
        self.spin_dwell.valueChanged.connect(lambda _v: _persist_light())
        self.edit_part_id.textChanged.connect(self._persist_part_id)

        # Front Inspections gallery placeholder
        self.group_gallery = QGroupBox("Front Inspections")
        gallery_layout = QVBoxLayout(self.group_gallery)
        self.gallery_placeholder = QLabel("No inspections yet.")
        self.gallery_placeholder.setAlignment(Qt.AlignCenter)
        gallery_layout.addWidget(self.gallery_placeholder)
        layout.addWidget(self.group_gallery, stretch=1)

        # Log
        self.group_log = QGroupBox("Log")
        log_layout = QVBoxLayout(self.group_log)
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        layout.addWidget(self.group_log)

        # Detection Results
        self.group_results = QGroupBox("Detection Results")
        res_layout = QVBoxLayout(self.group_results)
        self.results_table = QTableWidget(0, 7)
        self.results_table.setHorizontalHeaderLabels([
            "#",
            "Class",
            "Score",
            "Angle (deg)",
            "Phi (rad)",
            "Center (x,y)",
            "Bounds (x,y,w,h)",
        ])
        self.results_table.setEditTriggers(self.results_table.NoEditTriggers)
        self.results_table.setSelectionBehavior(self.results_table.SelectRows)
        self.results_table.setSelectionMode(self.results_table.SingleSelection)
        res_layout.addWidget(self.results_table)
        layout.addWidget(self.group_results)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(scroll)

    # Public helpers
    def set_attachment_loaded(self, loaded: bool):
        self._apply_primary_style(self.bt_attach_load, active=loaded)

    def set_front_loaded(self, loaded: bool):
        self._apply_primary_style(self.bt_front_load, active=loaded)

    def set_defect_loaded(self, loaded: bool):
        self._apply_primary_style(self.bt_defect_load, active=loaded)

    def _apply_primary_style(self, btn: QPushButton, active: bool):
        if active:
            btn.setStyleSheet(
                "QPushButton {"
                " background-color: #2e7d32; color: white;"
                " border: 2px solid #2e7d32; padding: 6px 10px; font-weight: 600;"
                "}"
                "QPushButton:hover { background-color: #388e3c; }"
            )
        else:
            btn.setStyleSheet(
                "QPushButton {"
                " background: transparent; color: #2e7d32;"
                " border: 2px solid #2e7d32; padding: 6px 10px; font-weight: 600;"
                "}"
                "QPushButton:hover { background-color: rgba(46,125,50,0.08); }"
            )

    # Prefill file paths from saved state
    def set_selected_files(self, attachment: str = None, front: str = None, defect: str = None):
        if attachment:
            self.edit_attach.setPlainText(attachment)
        if front:
            self.edit_front.setPlainText(front)
        if defect:
            self.edit_defect.setPlainText(defect)

    def append_log(self, text: str):
        self.log_text.appendPlainText(text)

    def populate_detection_results(self, detections):
        # detections is expected to be a list of dicts
        rows = []
        try:
            rows = sorted(detections or [], key=lambda d: int(d.get("index", 0)) or 0)
        except Exception:
            rows = list(detections or [])
        self.results_table.setRowCount(0)
        for det in rows:
            idx_val = det.get("index")
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            # Prefer bbox center if available; fallback to image center
            ctr = det.get("det_center", det.get("center", ""))
            values = [
                str(idx_val if idx_val is not None else ""),
                str(det.get("class", "")),
                f"{det.get('score', '')}",
                f"{det.get('angle', '')}",
                (f"{det.get('phi', 0.0):.3f}" if isinstance(det.get('phi', None), (int, float)) else ""),
                str(ctr),
                str(det.get("bounds", "")),
            ]
            for col, val in enumerate(values):
                self.results_table.setItem(row, col, QTableWidgetItem(val))

    # (no backend switching)

    def part_id(self) -> str:
        try:
            return self.edit_part_id.text().strip()
        except Exception:
            return ""

    def _persist_part_id(self):
        try:
            st = _state()
            st.part_id = self.edit_part_id.text().strip()
            _save_state()
        except Exception:
            pass
