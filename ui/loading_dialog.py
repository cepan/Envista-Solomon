from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar


class LoadingDialog(QDialog):
    def __init__(self, message: str = "Loading Envista Turntable…", parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setModal(False)
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setFixedSize(360, 120)
        self.setWindowTitle("")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)

        self.message = QLabel(message)
        self.message.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.message)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # marquee
        layout.addWidget(self.progress)

