# Python UI Export (Detectron2 + PyQt5)

PyQt5 translation of the DemoApp UI that runs Detectron2 models directly (no .NET/SolVision dependency). The app drives MindVision/iRAYPLE cameras, a turntable, and a linear axis, then runs a 4-step inspection flow with Detectron2 checkpoints.

## Features
- PyQt5 UI with workflow/logic tabs, preview panes, and defect ledger.
- Detectron2 inference for top/front/defect models (Mask R-CNN ResNet-101-FPN, single class `attachment` by default).
- Hardware control for iRAYPLE cameras, serial turntable, and linear axis; light controller settings are persisted.
- Captures organized under `captures/<PartID>/<YYYY-MM-DD>/<HHMMSS>/` with step-01 through step-04 outputs and cycle time logs.
- `data_extractor.py` flattens step-01/step-02 outputs into `captures_extracted/` for review or training.

## Prerequisites
- Windows, Python 3.12 (existing env: `envistaEnv12`) or Python 3.8 (`envistaEnv`).
- Packages: `PyQt5`, `numpy`, `opencv-python`, `torch`, `detectron2`, `pyserial`.
- MindVision/iRAYPLE SDK installed (Python files + runtime):
  - Headers: `C:\Program Files\HuarayTech\MV Viewer\Development\Samples\Python\IMV\MVSDK\IMVApi.py` and `IMVDefines.py`
  - Runtime: `C:\Program Files\HuarayTech\MV Viewer\Runtime\x64\MVSDKmd.dll`
  - Update paths in `services/camera_backend_irayple.py` if installed elsewhere.

## Setup
1. Create or activate a virtual environment (or reuse `envistaEnv12`):
   - `python -m venv .venv` then `.venv\Scripts\activate`
   - or `envistaEnv12\Scripts\activate`
2. Install dependencies (choose a Detectron2 wheel that matches your PyTorch/CUDA):
   - `pip install PyQt5 numpy opencv-python torch detectron2 pyserial`
3. Place your Detectron2 checkpoints (`.pth`) for attachment/top, front, and defect models somewhere accessible.

## Run the app
- From this folder: `python main.py`
- Wizard runs first to pick cameras/ports; the main window follows.
- Load models under "Step 1 - Selected Models":
  - Attachment = top model
  - Front Attachment = front-view model
  - Defect = defect classifier/detector on front crops
- Click **Run Detection**:
  - Step 1: capture from top camera (or loaded image), run Detectron2, compute arrows/indices.
  - Step 2: square crops saved to `step-02` using the configured size.
  - Step 3: front model runs on crops, saves annotated images and bbox crops.
  - Step 4: defect model runs on step-03 bbox crops, saves annotated images.
- GPU is used when available (`torch.cuda.is_available()`), otherwise CPU.

## Data and config
- Captures: `captures/<PartID>/<YYYY-MM-DD>/<HHMMSS>/` (raws, crops, front/defect outputs, `cycle_time.txt`).
- Flatten captures: `python data_extractor.py` -> `captures_extracted/`.
- UI state persists to `user_settings.json` (model paths, camera selections, light settings, crop size, etc.).

## Notable modules
- `main.py` - entry point; boots Detectron2 and launches the PyQt UI.
- `services/solvision_manager.py` - Detectron2 loading/inference helpers (single class list `CLASS_NAMES`).
- `services/camera_backend_irayple.py` / `services/camera_service.py` - MindVision camera backend.
- `services/turntable_service.py` / `services/linear_axis_service.py` - serial control for motion hardware.
- `ui/` - tabs, preview panel, tuner dialog, defect ledger, camera/turntable/axis panels.
- `data_extractor.py` - utility to copy step-01/step-02 outputs into `captures_extracted/`.

## Tips
- Models default to the `attachment` class; adjust `CLASS_NAMES` in `services/solvision_manager.py` if needed.
- If camera enumeration fails, confirm the iRAYPLE SDK paths and runtime are installed.
- Captures and `user_settings.json` are ignored by git (see `.gitignore`); keep model checkpoints out of version control.
