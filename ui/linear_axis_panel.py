from typing import Optional

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


class LinearAxisPanel(QWidget):
    refresh_requested = pyqtSignal()
    connect_requested = pyqtSignal(str)
    disconnect_requested = pyqtSignal()
    calibrate_requested = pyqtSignal()
    home_requested = pyqtSignal(float)
    goto_requested = pyqtSignal(float)
    port_selected = pyqtSignal(str)
    home_set_requested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        group = QGroupBox("Linear Axis (Front Camera)")
        root = QVBoxLayout(self)
        root.addWidget(group)

        v = QVBoxLayout(group)

        # Connection row
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

        v.addLayout(row)

        # Calibration / home row
        cal_row = QHBoxLayout()
        self.bt_calibrate = QPushButton("Calibrate")
        self.bt_calibrate.clicked.connect(self.calibrate_requested.emit)
        cal_row.addWidget(self.bt_calibrate)

        self.bt_home = QPushButton("Home (50 mm)")
        self.bt_home.clicked.connect(self._on_home)
        cal_row.addWidget(self.bt_home)

        v.addLayout(cal_row)

        # Position row
        pos_row = QHBoxLayout()
        pos_row.addWidget(QLabel("Position (mm):"))
        self.pos_spin = QDoubleSpinBox()
        self.pos_spin.setDecimals(1)
        self.pos_spin.setRange(0.0, 100.0)
        self.pos_spin.setSingleStep(1.0)
        self.pos_spin.setValue(50.0)
        pos_row.addWidget(self.pos_spin)

        self.bt_goto = QPushButton("Go")
        self.bt_goto.clicked.connect(self._on_goto)
        pos_row.addWidget(self.bt_goto)

        v.addLayout(pos_row)

        # Home setter reuses position box
        home_row = QHBoxLayout()
        self.bt_set_home = QPushButton("Set Home to Position")
        self.bt_set_home.clicked.connect(self._on_set_home)
        home_row.addWidget(self.bt_set_home)
        v.addLayout(home_row)

        self.status = QLabel("Disconnected.")
        v.addWidget(self.status)

        self._connected = False
        self._calibrated = False
        self._ready = False
        self._calibrating = False
        self._home_mm = 50.0
        self._update_enabled()

    def _on_toggle_connect(self):
        if self._connected:
            self.disconnect_requested.emit()
        else:
            port = self.port_combo.currentText().strip()
            if port:
                self.connect_requested.emit(port)

    def _on_home(self):
        self.home_requested.emit(float(self._home_mm))

    def _on_goto(self):
        self.goto_requested.emit(float(self.pos_spin.value()))

    def _on_set_home(self):
        mm = float(self.pos_spin.value())
        self.set_home_mm(mm)
        self.home_set_requested.emit(mm)

    def set_calibrating(self, calibrating: bool):
        self._calibrating = bool(calibrating)
        if self._calibrating:
            self.set_status("Calibrating...")
        self._update_enabled()
        if not self._calibrating and self._calibrated:
            self.set_status(f"Calibrated. Current position: {self.pos_spin.value():.1f} mm.")

    def set_ports(self, ports):
        self.port_combo.blockSignals(True)
        self.port_combo.clear()
        for p in ports:
            self.port_combo.addItem(p)
        self.port_combo.blockSignals(False)

    def set_connected(self, connected: bool, port: Optional[str] = None):
        self._connected = connected
        self._ready = False if connected else False
        self._update_enabled()
        self.bt_connect.setText("Disconnect" if connected else "Connect")
        self._apply_connect_style(connected)
        if connected:
            self.set_status(f"Connected ({port or ''}). Waiting for controller...")
        else:
            self.set_status("Disconnected.")

    def set_calibrated(self, calibrated: bool, position_mm: Optional[float] = None):
        self._calibrated = calibrated
        self._calibrating = False
        if calibrated:
            self._ready = True
        self._update_enabled()
        if calibrated:
            if position_mm is not None:
                self.pos_spin.setValue(float(position_mm))
            self.set_status(f"Calibrated. Current position: {self.pos_spin.value():.1f} mm.")
        else:
            self.set_status("Connected but not calibrated.")

    def set_position(self, position_mm: float):
        try:
            self.pos_spin.setValue(float(position_mm))
        except Exception:
            pass

    def set_ready(self, ready: bool):
        self._ready = bool(ready)
        self._update_enabled()
        if self._connected:
            if self._ready:
                self.set_status(f"Controller ready. Position: {self.pos_spin.value():.1f} mm.")
            else:
                self.set_status("Connected. Waiting for controller...")

    def set_status(self, text: str):
        self.status.setText(text or "")

    def set_home_mm(self, mm: float):
        try:
            self._home_mm = float(mm)
            self._update_home_button_text(mm)
        except Exception:
            pass

    def home_mm(self) -> float:
        return float(self._home_mm)

    def is_calibrating(self) -> bool:
        return bool(self._calibrating)

    def is_ready(self) -> bool:
        return bool(self._ready)

    def _update_enabled(self):
        connected = self._connected
        busy = self._calibrating

        # Do not allow changing ports or reconnecting while a calibration is running
        self.port_combo.setEnabled(not connected and not busy)
        self.bt_refresh.setEnabled(not connected and not busy)
        self.bt_connect.setEnabled(not busy)

        # Calibrate requires controller ready, connection, and no ongoing calibration
        can_calibrate = connected and self._ready and not busy
        self.bt_calibrate.setEnabled(can_calibrate)

        # Home/Go/position require calibration and should stay disabled while calibrating
        can_move = connected and self._calibrated and not busy
        self.bt_home.setEnabled(can_move)
        self.pos_spin.setEnabled(can_move)
        self.bt_goto.setEnabled(can_move)
        # Home config requires controller ready (to avoid pre-banner clicks)
        can_configure_home = connected and self._ready and not busy
        self.bt_set_home.setEnabled(can_configure_home)
        # Keep connect button style in sync after enable/disable changes
        self._apply_connect_style(connected)

        # Calibrate button color: red while calibrating, green when calibrated, default otherwise
        if self._calibrating:
            self.bt_calibrate.setStyleSheet(
                "QPushButton {"
                " background-color: #c62828; color: white;"
                " border: 2px solid #c62828; padding: 6px 10px; font-weight: 600;"
                "}"
                "QPushButton:hover { background-color: #d32f2f; }"
            )
        elif self._calibrated:
            self.bt_calibrate.setStyleSheet(
                "QPushButton {"
                " background-color: #2e7d32; color: white;"
                " border: 2px solid #2e7d32; padding: 6px 10px; font-weight: 600;"
                "}"
                "QPushButton:hover { background-color: #388e3c; }"
            )
        else:
            self.bt_calibrate.setStyleSheet(
                "QPushButton {"
                " background: transparent; color: #2e7d32;"
                " border: 2px solid #2e7d32; padding: 6px 10px; font-weight: 600;"
                "}"
                "QPushButton:hover { background-color: rgba(46,125,50,0.08); }"
            )

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

    def _update_home_button_text(self, mm: float):
        try:
            self.bt_home.setText(f"Home ({float(mm):.1f} mm)")
        except Exception:
            pass
