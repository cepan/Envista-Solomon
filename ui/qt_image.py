from typing import Optional
import numpy as np
from PyQt5.QtGui import QImage, QPixmap


def np_bgr_to_qpixmap(arr: np.ndarray) -> Optional[QPixmap]:
    if arr is None or arr.ndim != 3 or arr.shape[2] != 3:
        return None
    h, w, _ = arr.shape
    # Convert BGR -> RGB
    rgb = arr[:, :, ::-1].copy()
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

