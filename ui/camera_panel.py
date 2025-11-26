from typing import List, Dict, Optional
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QComboBox,
    QPushButton,
)


class _RoleWidget(QWidget):
    selection_changed = pyqtSignal(str, int)
    connect_clicked = pyqtSignal(str)
    capture_clicked = pyqtSignal(str)

    def __init__(self, role: str, header: str, parent=None):
        super().__init__(parent)
        self.role = role
        self._connected: bool = False
        self._connected_index: Optional[int] = None

        layout = QVBoxLayout(self)

        self.header_label = QLabel(header)
        layout.addWidget(self.header_label)

        self.detail_label = QLabel("Choose which device should act as the camera.")
        layout.addWidget(self.detail_label)

        self.selector = QComboBox()
        self.selector.currentIndexChanged.connect(self._on_selection_changed)
        layout.addWidget(self.selector)

        row = QHBoxLayout()
        self.bt_connect = QPushButton("Connect Camera")
        self.bt_connect.clicked.connect(self._on_connect)
        self.bt_connect.setEnabled(False)
        self._apply_connect_style(False)
        row.addWidget(self.bt_connect)
        self.bt_capture = QPushButton("Capture Preview")
        self.bt_capture.clicked.connect(self._on_capture)
        self.bt_capture.setEnabled(False)
        row.addWidget(self.bt_capture)
        layout.addLayout(row)

        self.status_label = QLabel("Camera not connected.")
        layout.addWidget(self.status_label)

    def _on_selection_changed(self, _idx: int):
        data = self.selector.currentData()
        if isinstance(data, dict) and "index" in data:
            self.selection_changed.emit(self.role, int(data["index"]))
        self._refresh_detail()

    def _on_connect(self):
        self.connect_clicked.emit(self.role)

    def _on_capture(self):
        self.capture_clicked.emit(self.role)

    def set_devices(self, devices: List[Dict]):
        current = self.selector.currentData()
        self.selector.blockSignals(True)
        self.selector.clear()
        for d in devices:
            self.selector.addItem(d.get("name", f"Camera {d.get('index','?')}") , d)
        # Try to restore selection
        if current is not None:
            for i in range(self.selector.count()):
                if self.selector.itemData(i) == current:
                    self.selector.setCurrentIndex(i)
                    break
        self.selector.blockSignals(False)
        self._refresh_detail()

    def selected_device_index(self) -> Optional[int]:
        data = self.selector.currentData()
        if isinstance(data, dict) and "index" in data:
            return int(data["index"])
        return None

    def set_connected(self, connected: bool, device_name: Optional[str] = None):
        self._connected = connected
        self.bt_connect.setText("Disconnect" if connected else "Connect Camera")
        self.bt_connect.setEnabled(connected or self.selected_device_index() is not None)
        self.bt_capture.setEnabled(connected)
        self._apply_connect_style(connected)
        if connected and device_name:
            self.status_label.setText(f"Connected ({device_name}).")
        else:
            self.status_label.setText("Camera not connected.")
        self._refresh_detail()

    def _refresh_detail(self):
        has_selection = self.selected_device_index() is not None
        # Connect button enabled if connected or a selection exists
        self.bt_connect.setEnabled(self._connected or has_selection)
        if self._connected:
            name = self.selector.currentText() or "device"
            self.detail_label.setText(f"{name}\nCapture a frame to refresh the preview.")
        else:
            if has_selection:
                name = self.selector.currentText()
                self.detail_label.setText(
                    f"Selected: {name}\nClick Connect Camera to assign it as the {self.role.lower()} view."
                )
            else:
                self.detail_label.setText(
                    f"Choose which device should act as the {self.role.lower()} camera."
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

    # Programmatic selection helper
    def select_device_index(self, index: int):
        for i in range(self.selector.count()):
            data = self.selector.itemData(i)
            if isinstance(data, dict) and int(data.get("index", -1)) == int(index):
                self.selector.setCurrentIndex(i)
                break


class CameraPanel(QWidget):
    refresh_requested = pyqtSignal()
    connect_requested = pyqtSignal(str, int)
    disconnect_requested = pyqtSignal(str)
    capture_requested = pyqtSignal(str)
    selection_changed = pyqtSignal(str, int)

    def __init__(self, parent=None):
        super().__init__(parent)

        root = QVBoxLayout(self)
        group = QGroupBox("Step 2 - Configure Cameras")
        root.addWidget(group)

        v = QVBoxLayout(group)

        # Toolbar
        toolbar = QHBoxLayout()
        self.bt_refresh = QPushButton("Refresh Devices")
        self.bt_refresh.clicked.connect(self.refresh_requested)
        toolbar.addWidget(self.bt_refresh)
        hint = QLabel("Assign top and front cameras, then capture a preview to verify the feed.")
        toolbar.addWidget(hint)
        toolbar.addStretch(1)
        v.addLayout(toolbar)

        # Role widgets (Top / Front)
        self.top = _RoleWidget("Top", "Top Camera")
        self.front = _RoleWidget("Front", "Front Camera")

        self.top.connect_clicked.connect(self._on_connect_clicked)
        self.front.connect_clicked.connect(self._on_connect_clicked)
        self.top.capture_clicked.connect(self.capture_requested)
        self.front.capture_clicked.connect(self.capture_requested)
        self.top.selection_changed.connect(self.selection_changed)
        self.front.selection_changed.connect(self.selection_changed)

        v.addWidget(self.top)
        v.addWidget(self.front)

    # External interface
    def set_devices(self, devices: List[Dict]):
        self.top.set_devices(devices)
        self.front.set_devices(devices)

    def set_connected(self, role: str, connected: bool, device_name: Optional[str] = None):
        if role == "Top":
            self.top.set_connected(connected, device_name)
        else:
            self.front.set_connected(connected, device_name)

    def selected_index(self, role: str) -> Optional[int]:
        return self.top.selected_device_index() if role == "Top" else self.front.selected_device_index()

    def set_selected_index(self, role: str, index: int):
        if role == "Top":
            self.top.select_device_index(index)
        else:
            self.front.select_device_index(index)

    # Internals
    def _on_connect_clicked(self, role: str):
        widget = self.top if role == "Top" else self.front
        if widget._connected:
            self.disconnect_requested.emit(role)
        else:
            idx = widget.selected_device_index()
            if idx is not None:
                self.connect_requested.emit(role, idx)
