from __future__ import annotations

import faulthandler
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

# Crash log lives beside main.py so it is easy to find.
LOG_PATH = Path(__file__).resolve().parents[1] / "crash.log"

_orig_sys_hook = None
_orig_thread_hook = None
_orig_qt_handler = None
_faulthandler_file = None


def _write_line(text: str) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception:
        # Never raise from crash logging.
        pass


def _log_exception(prefix: str, exc_type, exc_value, exc_tb) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    header = f"=== {prefix} @ {ts} ==="
    _write_line(header)
    try:
        tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        _write_line(tb_text.rstrip())
    except Exception:
        _write_line(f"[crash_reporter] failed to format traceback for {exc_type}")
    _write_line("")


def _maybe_show_dialog(summary: str) -> None:
    try:
        from PyQt5.QtWidgets import QApplication, QMessageBox

        app = QApplication.instance()
        if app is None:
            return
        QMessageBox.critical(
            None,
            "Unexpected Error",
            f"{summary}\n\nDetails were written to:\n{LOG_PATH}",
        )
    except Exception:
        # Avoid secondary crashes when trying to notify the user.
        pass


def _sys_excepthook(exc_type, exc_value, exc_tb) -> None:
    _log_exception("Unhandled exception", exc_type, exc_value, exc_tb)
    _maybe_show_dialog(f"{exc_type.__name__}: {exc_value}")
    if callable(_orig_sys_hook):
        _orig_sys_hook(exc_type, exc_value, exc_tb)


def _thread_excepthook(args) -> None:
    try:
        _log_exception("Unhandled thread exception", args.exc_type, args.exc_value, args.exc_traceback)
    finally:
        if callable(_orig_thread_hook):
            _orig_thread_hook(args)


def _qt_message_handler(mode, context, message) -> None:
    # mode is QtMsgType; map to readable labels.
    prefix = getattr(mode, "name", None) or str(mode)
    try:
        _write_line(f"[Qt/{prefix}] {message}")
    finally:
        if callable(_orig_qt_handler):
            try:
                _orig_qt_handler(mode, context, message)
            except Exception:
                pass


def install(log_path: Optional[Path] = None) -> None:
    """Install crash hooks to capture unexpected exceptions and Qt messages."""
    global LOG_PATH, _orig_sys_hook, _orig_thread_hook, _orig_qt_handler, _faulthandler_file

    if log_path is not None:
        LOG_PATH = Path(log_path)

    # Enable Python's faulthandler to capture hard crashes (segfaults).
    try:
        _faulthandler_file = LOG_PATH.open("a", encoding="utf-8")
        faulthandler.enable(_faulthandler_file, all_threads=True)
    except Exception:
        _faulthandler_file = None

    # Hook sys.excepthook for main thread crashes.
    _orig_sys_hook = sys.excepthook
    sys.excepthook = _sys_excepthook

    # Hook threading.excepthook (Python 3.8+) for background thread crashes.
    if hasattr(threading, "excepthook"):
        _orig_thread_hook = threading.excepthook
        threading.excepthook = _thread_excepthook

    # Capture Qt warnings/errors; optional so we don't fail if PyQt is absent.
    try:
        from PyQt5.QtCore import qInstallMessageHandler

        _orig_qt_handler = qInstallMessageHandler(_qt_message_handler)
    except Exception:
        _orig_qt_handler = None
