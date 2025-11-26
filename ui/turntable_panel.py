from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QWidget,
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QPushButton,
    QLabel,
    QDoubleSpinBox,
)


class TurntablePanel(QWidget):
    refresh_requested = pyqtSignal()
    connect_requested = pyqtSignal(str)
    disconnect_requested = pyqtSignal()
    home_requested = pyqtSignal()
    rotate_requested = pyqtSignal(float)
    port_selected = pyqtSignal(str)
    step_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        group = QGroupBox("Step 3 - Prepare Turntable")
        root = QVBoxLayout(self)
        root.addWidget(group)

        v = QVBoxLayout(group)

        row = QHBoxLayout()
        self.port_combo = QComboBox()
        self.port_combo.currentTextChanged.connect(lambda s: self.port_selected.emit(s))
        row.addWidget(self.port_combo, stretch=1)

        self.bt_refresh = QPushButton("Refresh")
        self.bt_refresh.clicked.connect(self.refresh_requested.emit)
        row.addWidget(self.bt_refresh)

        self.bt_connect = QPushButton("Connect")
        self._apply_connect_style(False)
        self.bt_connect.clicked.connect(self._on_toggle_connect)
        row.addWidget(self.bt_connect)

        self.bt_home = QPushButton("Home")
        self.bt_home.clicked.connect(self.home_requested.emit)
        row.addWidget(self.bt_home)

        v.addLayout(row)

        # Rotate controls
        rot_row = QHBoxLayout()
        self.step = QDoubleSpinBox()
        self.step.setDecimals(2)
        self.step.setRange(-3600.0, 3600.0)
        self.step.setSingleStep(5.0)
        self.step.setValue(45.0)
        self.step.valueChanged.connect(lambda v: self.step_changed.emit(float(v)))
        rot_row.addWidget(QLabel("Step (deg):"))
        rot_row.addWidget(self.step)
        self.bt_ccw = QPushButton("Rotate -")
        self.bt_ccw.clicked.connect(lambda: self.rotate_requested.emit(-abs(self.step.value())))
        rot_row.addWidget(self.bt_ccw)
        self.bt_cw = QPushButton("Rotate +")
        self.bt_cw.clicked.connect(lambda: self.rotate_requested.emit(abs(self.step.value())))
        rot_row.addWidget(self.bt_cw)
        v.addLayout(rot_row)

        self.status = QLabel("Disconnected.")
        v.addWidget(self.status)

        self._connected = False

    def _on_toggle_connect(self):
        if self._connected:
            self.disconnect_requested.emit()
        else:
            port = self.port_combo.currentText().strip()
            if port:
                self.connect_requested.emit(port)

    def set_ports(self, ports):
        self.port_combo.blockSignals(True)
        self.port_combo.clear()
        for p in ports:
            self.port_combo.addItem(p)
        self.port_combo.blockSignals(False)

    def set_connected(self, connected: bool, port: str = None):
        self._connected = connected
        self.port_combo.setEnabled(not connected)
        self.bt_refresh.setEnabled(not connected)
        self.bt_connect.setText("Disconnect" if connected else "Connect")
        self._apply_connect_style(connected)
        self.bt_home.setEnabled(connected)
        self.bt_ccw.setEnabled(connected)
        self.bt_cw.setEnabled(connected)
        self.step.setEnabled(connected)
        if connected:
            self.set_status(f"Connected ({port or ''}).")
        else:
            self.set_status("Disconnected.")

    def set_status(self, text: str):
        # Convert any scientific notation numbers to fixed decimals for readability
        import re
        def repl(m):
            try:
                return f"{float(m.group(0)):.2f}"
            except Exception:
                return m.group(0)
        cleaned = re.sub(r"[-+]?\d*\.?\d+[eE][-+]?\d+", repl, text or "")
        self.status.setText(cleaned)

    def _apply_connect_style(self, connected: bool):
        if connected:
            self.bt_connect.setStyleSheet(
                "QPushButton {"
                " background-color: #2e7d32; color: white;"
                " border: 2px solid #2e7d32; padding: 6px 10px; font-weight: 600;"
                "}"
                "QPushButton:hover { background-color: #388e3c; }"
            )
        else:
            self.bt_connect.setStyleSheet(
                "QPushButton {"
                " background: transparent; color: #2e7d32;"
                " border: 2px solid #2e7d32; padding: 6px 10px; font-weight: 600;"
                "}"
                "QPushButton:hover { background-color: rgba(46,125,50,0.08); }"
            )
