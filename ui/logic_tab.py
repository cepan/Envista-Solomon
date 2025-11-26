from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel


class LogicTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Logic Builder placeholder. To be implemented."))

