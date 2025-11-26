from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QTableWidget, QTableWidgetItem


class DefectLedger(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("Defect Ledger")
        v = QVBoxLayout(group)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([
            "Index",
            "Class",
            "Confidence",
            "Area",
            "Bounds (x,y,w,h)",
        ])
        self.table.setEditTriggers(self.table.NoEditTriggers)
        self.table.setSelectionBehavior(self.table.SelectRows)
        self.table.setSelectionMode(self.table.SingleSelection)
        v.addWidget(self.table)

        layout.addWidget(group)

    def populate_ledger(self, detections):
        # detections is expected to be a list of dicts
        self.table.setRowCount(0)
        for idx, det in enumerate(detections, start=1):
            row = self.table.rowCount()
            self.table.insertRow(row)
            values = [
                str(idx),
                str(det.get("class", "")),
                f"{det.get('score', '')}",
                str(det.get("area", "")),
                str(det.get("bounds", "")),
            ]
            for col, val in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(val))

