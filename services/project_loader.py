from . import solvision_manager
from .config import state, save_state


def load_project(path: str):
    """Load a Detectron2 model checkpoint for the default (top) role."""
    if not solvision_manager.ensure_initialized():
        raise RuntimeError(solvision_manager.initialization_error() or "Model initialization failed")
    solvision_manager.load_project(path)
    # Persist last project path
    st = state(); st.last_project_path = path; save_state()
    return {"path": path}
