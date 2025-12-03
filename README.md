# Envista-Solomon

**Automated Visual Inspection System with Detectron2 AI and Multi-Axis Robotics**

A PyQt5-based industrial inspection application that combines Detectron2 deep learning models with robotic hardware control for automated multi-angle part inspection. The system captures images from multiple camera perspectives, detects objects and defects using AI, and orchestrates turntable rotation and linear axis positioning for comprehensive 360-degree inspection.

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)
[![Detectron2](https://img.shields.io/badge/AI-Detectron2-orange.svg)](https://github.com/facebookresearch/detectron2)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

---

## Screenshots

> **Note**: Add actual screenshots to a `docs/images/` folder and update these placeholders

### Main Application Window
![Main Window](docs/images/main_window.png)
*Main interface showing workflow controls (left) and dual-pane preview with overlays (right)*

### Initialization Wizard
![Init Wizard](docs/images/init_wizard.png)
*Step-by-step setup wizard for hardware configuration*

### Detection Results
![Detection Results](docs/images/detection_results.png)
*Top view with AI-detected attachments, arrows, and CCW indices*

### Defect Analysis
![Defect Detection](docs/images/defect_detection.png)
*Defect classification with confidence scores and annotations*

### Edge Tuner Dialog
![Edge Tuner](docs/images/edge_tuner.png)
*Interactive parameter tuning for contour detection*

---

## Table of Contents
- [Features](#features)
- [System Overview](#system-overview)
- [Hardware Requirements](#hardware-requirements)
- [Software Prerequisites](#software-prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Configuration](#configuration)
- [Data Management](#data-management)
- [Building Windows Executable](#building-windows-executable)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Features

### 🖥️ User Interface
- **Modern PyQt5 GUI** with tabbed workflow interface
- **Dual-pane image preview** with real-time overlay visualization
- **Interactive edge/contour tuner** for detection parameter adjustment
- **Setup wizard** for guided hardware initialization
- **Live detection table** with scores, angles, and bounding boxes
- **Defect ledger** for tracking quality issues
- **Comprehensive logging** with timestamped events

### 🤖 AI-Powered Detection
- **Detectron2 integration** using Mask R-CNN ResNet-101-FPN architecture
- **Three-stage detection pipeline**:
  - **Stage 1**: Top-view attachment detection with orientation analysis
  - **Stage 2**: Front-view alignment and capture
  - **Stage 3**: Defect classification with configurable thresholds
- **GPU acceleration** with automatic CPU fallback
- **Custom model support** via Detectron2 checkpoints

### 🔧 Hardware Control
- **Multi-backend camera support**:
  - iRAYPLE/MindVision industrial cameras
  - Basler cameras (via Pylon SDK)
- **Automated turntable control** via serial communication
- **Linear axis positioning** (0-100mm travel) with auto-calibration
- **LED light controller** (ULC-2) with programmable brightness per camera
- **Synchronized multi-axis motion** for precise alignment

### 📊 Advanced Inspection Workflow
- **4-step automated inspection pipeline**:
  1. Top camera capture → AI detection → Arrow/angle computation
  2. Turntable rotation + Linear axis positioning → Front capture
  3. Front-view AI detection → Bounding box extraction
  4. Defect AI analysis → Quality classification
- **Parallel processing** with background inference
- **Automatic FOV-based alignment** using calibrated pixel-to-mm conversion
- **Cycle time tracking** for performance optimization

### 💾 Data Management
- **Organized capture structure**: `captures/<PartID>/<Date>/<Time>/step-XX/`
- **Comprehensive image outputs**: Raw, annotated, cropped, and bbox variants
- **Data extraction tool** for flattening hierarchical captures
- **Persistent configuration** with JSON-based state management
- **Crash logging** with full traceback capture

---

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Envista-Solomon                          │
│                   PyQt5 Main Window                          │
├─────────────────┬───────────────────────────────────────────┤
│  Workflow Tab   │  Image Preview Panel                      │
│                 │  ┌─────────────┬─────────────┐            │
│ ┌─────────────┐ │  │ Top View    │ Front View  │            │
│ │ Model Setup │ │  │ (Overlays)  │ (Crops)     │            │
│ ├─────────────┤ │  └─────────────┴─────────────┘            │
│ │Camera Panel │ │                                            │
│ ├─────────────┤ │  Defect Ledger                            │
│ │Turntable Ctl│ │  ┌──────────────────────────┐             │
│ ├─────────────┤ │  │ Index │ Class │ Score    │             │
│ │Linear Axis  │ │  └──────────────────────────┘             │
│ ├─────────────┤ │                                            │
│ │Light Control│ │                                            │
│ ├─────────────┤ │                                            │
│ │Run Detection│ │                                            │
│ └─────────────┘ │                                            │
└─────────────────┴───────────────────────────────────────────┘
          │
          ├──► Camera Manager ──► iRAYPLE/Pylon Backend
          ├──► Turntable Service ──► Serial Controller
          ├──► Linear Axis Service ──► Serial Controller
          ├──► Light Controller ──► ULC-2 (UDP)
          └──► Solvision Manager ──► Detectron2 Inference
```

---

## Hardware Requirements

### Required Components
| Component | Specification | Purpose |
|-----------|---------------|---------|
| **Computer** | Windows 10/11, 16GB+ RAM | Host system |
| **GPU** | NVIDIA GPU with CUDA 12.6 support | AI inference acceleration |
| **Top Camera** | Industrial camera (iRAYPLE/Basler) | Top-view part imaging |
| **Front Camera** | Industrial camera (iRAYPLE/Basler) | Front-view inspection |

### Optional Components
| Component | Specification | Purpose |
|-----------|---------------|---------|
| **Turntable** | Serial-controlled rotary stage | Part rotation for multi-angle capture |
| **Linear Axis** | FUYU 100mm travel linear stage | Camera/part positioning |
| **Light Controller** | ULC-2 LED controller | Programmable lighting control |

### Camera Hardware

#### Option 1: iRAYPLE/MindVision Cameras (Recommended)
- **Installation**: Download and install [MV Viewer](https://en.mindvision.com.cn/ziliaoxiazai/ruanjianxiazai/) from MindVision
- **Required files**:
  - SDK headers: `C:\Program Files\HuarayTech\MV Viewer\Development\Samples\Python\IMV\MVSDK\IMVApi.py` + `IMVDefines.py`
  - Runtime DLL: `C:\Program Files\HuarayTech\MV Viewer\Runtime\x64\MVSDKmd.dll`
- **Custom paths**: Edit `services/camera_backend_irayple.py` if installed elsewhere

#### Option 2: Basler Cameras
- **Installation**: Download [pypylon SDK](https://github.com/basler/pypylon)
- **Python package**: `pip install pypylon`
- **Compatibility**: Tested with Basler ace/dart series

### Turntable/Linear Axis
- **Communication**: Serial (USB-to-Serial adapter recommended)
- **Baud rates**:
  - Turntable: 115200 (8N1)
  - Linear Axis: 9600 (8N1)
- **Drivers**: Install [CH340/FTDI drivers](https://learn.sparkfun.com/tutorials/how-to-install-ch340-drivers/all) as needed

### LED Light Controller
- **Model**: ULC-2 or compatible
- **Protocol**: UDP over Ethernet
- **Network**: Static IP recommended (e.g., `192.168.1.100`)

---

## Software Prerequisites

### Operating System
- **Windows 10/11** (64-bit)
- **Linux** support experimental (camera SDKs may require manual setup)

### Python Environment
- **Python 3.8 - 3.12** (3.12 recommended for development)
- **Virtual environment** strongly recommended

### CUDA Toolkit
- **CUDA 12.6** runtime and drivers
- Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-12-6-0-download-archive)
- Verify installation: `nvidia-smi` (should show CUDA 12.6+)

### Python Dependencies
See [requirements.txt](requirements.txt) for full list. Key packages:
- `PyQt5` - GUI framework
- `torch==2.7.1` - Deep learning (CUDA 12.6 compatible)
- `detectron2==0.6+18f6958pt2.7.1cu126` - Object detection
- `opencv-python` - Image processing
- `numpy` - Numerical operations
- `pyserial` - Serial communication

---

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/Envista-Solomon.git
cd Envista-Solomon
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv envistaEnv12
envistaEnv12\Scripts\activate

# Linux/macOS
python3 -m venv envistaEnv12
source envistaEnv12/bin/activate
```

### 3. Install PyTorch (CUDA 12.6)
```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

### 4. Install Detectron2
```bash
pip install detectron2==0.6+18f6958pt2.7.1cu126 -f https://miropsota.github.io/torch_packages_builder/detectron2/
```

### 5. Install Remaining Dependencies
```bash
pip install -r requirements.txt
```

### 6. Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import detectron2; print(f'Detectron2: {detectron2.__version__}')"
```

Expected output:
```
PyTorch: 2.7.1+cu126
CUDA Available: True
Detectron2: 0.6+18f6958pt2.7.1cu126
```

### 7. Install Camera SDKs (Optional)
- **iRAYPLE**: Install MV Viewer from vendor
- **Basler**: `pip install pypylon`

---

## Quick Start

### First Run
```bash
python main.py
```

The **Initialization Wizard** will guide you through:
1. **Model Selection**: Browse to your Detectron2 checkpoint files (`.pth`)
2. **Camera Setup**: Connect top and front cameras
3. **Preview Test**: Verify camera captures
4. **Turntable Setup**: Connect and home the turntable

### Running an Inspection
1. **Load Models**: In the Workflow tab, browse to your trained models:
   - Attachment Model (top-view detection)
   - Front Attachment Model (front-view detection)
   - Defect Model (defect classification)

2. **Configure Hardware**:
   - Select cameras from dropdowns and click **Connect**
   - Select turntable port and click **Connect**
   - (Optional) Select linear axis port, **Connect**, then **Calibrate**

3. **Set Parameters**:
   - **Part ID**: Enter identifier for this inspection run
   - **Crop Size**: Adjust square crop size (default 1600px)
   - **Light Settings**: Configure brightness for each camera
   - **Defect Threshold**: Set minimum confidence (default 0.5)

4. **Run Detection**:
   - Click **Run Detection** to start the 4-step workflow
   - Monitor progress in the log panel
   - View results in the preview panes and detection table

### Outputs
All results are saved to:
```
captures/<PartID>/<YYYY-MM-DD>/<HHMMSS>/
  step-01_top_raw.png              # Raw top camera image
  step-01_top_annotated.png        # Annotated with detections
  step-01 cropped images/          # Individual detection crops
  step-02/                         # Front camera captures per detection
    step-02_front_initial_001.png
    step-02_front_crop_001.png
    step_2_cropped/                # Aligned front crops
  step-03/                         # Front detection results
    step-03_front_001.png
    step-03_front_bbox_001.png
  step-04/                         # Defect detection results
    step-04_defect_001.png
  cycle_time.txt                   # Total inspection time
```

---

## Detailed Usage

### Inspection Workflow Overview

The system performs automated multi-angle inspection using a 4-step pipeline that coordinates cameras, AI models, and robotic hardware:

```
                           ┌─────────────────────────────────────┐
                           │   ENVISTA-SOLOMON WORKFLOW          │
                           └─────────────────────────────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    │                                              │
              ┌─────▼─────┐                                 ┌──────▼──────┐
              │  STEP 1   │                                 │   CAMERAS   │
              │ Top View  │◄────────────────────────────────┤ Top + Front │
              │ Detection │                                 └─────────────┘
              └─────┬─────┘
                    │ Detects: Attachment 1 @ 45°
                    │          Attachment 2 @ 135°
                    │          Attachment 3 @ 270°
                    │
              ┌─────▼─────┐
              │  STEP 2   │                         ┌──────────────────┐
              │Multi-Angle│◄────────────────────────┤  TURNTABLE +     │
              │  Capture  │                         │  LINEAR AXIS     │
              └─────┬─────┘                         └──────────────────┘
                    │                                         ▲
                    │                                         │
                    ├──► For each attachment:                │
                    │    1. Rotate turntable to angle        │
                    │    2. Position linear axis       ──────┘
                    │    3. Capture front view
                    │    4. Move to next (while AI processes in background)
                    │
              ┌─────▼─────┐
              │  STEP 3   │                         ┌──────────────────┐
              │Front View │◄────────────────────────┤   DETECTRON2     │
              │ Detection │                         │  Front Model     │
              └─────┬─────┘                         └──────────────────┘
                    │ Extracts bbox of attachment in front view
                    │
              ┌─────▼─────┐
              │  STEP 4   │                         ┌──────────────────┐
              │  Defect   │◄────────────────────────┤   DETECTRON2     │
              │   Check   │                         │  Defect Model    │
              └─────┬─────┘                         └──────────────────┘
                    │ Classifies: Pass/Fail with confidence
                    │
              ┌─────▼─────┐
              │  RESULTS  │
              │  • Images │
              │  • Scores │
              │  • Report │
              └───────────┘
```

### Example Scenario: Inspecting a Circuit Board

**Part**: Circuit board with 4 connector attachments

**Step 1 - Top View Detection**:
```
Top Camera captures overhead view
AI Model detects 4 connectors:
  • Connector #1 at (x:850, y:600)  → phi: 0.785 rad (45°)
  • Connector #2 at (x:200, y:600)  → phi: 2.356 rad (135°)
  • Connector #3 at (x:200, y:1400) → phi: 3.927 rad (225°)
  • Connector #4 at (x:850, y:1400) → phi: 5.498 rad (315°)

System assigns CCW indices starting from bottom-right: #4, #1, #2, #3
```

**Step 2 - Multi-Angle Capture**:
```
Inspecting Connector #4 (bottom-right):
  ├─ Turntable rotates to 315° (5.498 rad)
  ├─ Linear axis moves to 62.5mm (centers connector in front camera)
  ├─ Front camera captures aligned view
  └─ Saves: step-02_front_001.png

Moving to Connector #1 (top-right):
  ├─ Turntable rotates to 45° (0.785 rad)
  ├─ Linear axis moves to 58.2mm
  ├─ Front camera captures aligned view
  └─ Saves: step-02_front_002.png

(Meanwhile, Step 3 analyzes Connector #4 in background...)

Repeats for Connectors #2 and #3
Total motion time: ~8 seconds (parallel processing saves ~15 seconds)
```

**Step 3 - Front View Detection**:
```
For each front view capture:
  ├─ AI detects the connector in the image
  ├─ Selects best detection (highest score, closest to center)
  ├─ Extracts bbox with 50px padding
  └─ Saves: step-03_front_bbox_001.png

Example detection:
  • Class: "connector"
  • Score: 0.94
  • Bbox: (245, 380, 420, 680)
```

**Step 4 - Defect Classification**:
```
For each connector bbox:
  ├─ AI inspects for defects
  ├─ Applies threshold (0.5)
  └─ Annotates findings

Connector #1: ✓ PASS (no defects)
Connector #2: ✓ PASS (no defects)
Connector #3: ✗ FAIL
  • Defect: "bent_pin" (score: 0.87)
  • Defect: "scratch" (score: 0.62)
Connector #4: ✓ PASS (no defects)

Total cycle time: 23.4 seconds
```

**Results Saved To**:
```
captures/CircuitBoard_Rev2/2025-12-03/143022/
  ├── step-01_top_raw.png              (overhead view)
  ├── step-01_top_annotated.png        (with arrows + indices)
  ├── step-02/step-02_front_001.png    (4 front captures)
  ├── step-03/step-03_front_bbox_001.png (4 bbox crops)
  ├── step-04/step-04_defect_003.png   (annotated defect on #3)
  └── cycle_time.txt                   (23.4 seconds)
```

### Visual Workflow Breakdown

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            STEP 1: TOP VIEW                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INPUT: Top camera capture                                              │
│  ┌─────────────────────────────────────────────────┐                   │
│  │              [Top Camera Image]                 │                   │
│  │                                                 │                   │
│  │     ●────►                    ●────►            │  ← AI detects     │
│  │    [1]                       [2]                │    attachments    │
│  │                                                 │    & computes     │
│  │                                                 │    arrows/angles  │
│  │           ●────►        ●────►                  │                   │
│  │          [4]           [3]                      │                   │
│  └─────────────────────────────────────────────────┘                   │
│                                                                         │
│  OUTPUT: Annotated image + 4 detection crops                           │
│  DATA: [(x,y), phi_angle, bbox, score] for each detection              │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      STEP 2: MULTI-ANGLE CAPTURE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  FOR EACH DETECTION (CCW order: 4→1→2→3):                              │
│                                                                         │
│  ┌──────────────────┐    ┌──────────────────┐                         │
│  │   TURNTABLE      │    │   LINEAR AXIS    │                         │
│  │   Rotate to      │◄───┤   Move to        │◄─── Calculated from     │
│  │   phi angle      │    │   X-position     │     detection center    │
│  └────────┬─────────┘    └────────┬─────────┘                         │
│           │                       │                                    │
│           └───────────┬───────────┘                                    │
│                       ▼                                                │
│           ┌────────────────────────┐                                   │
│           │   FRONT CAMERA         │                                   │
│           │   Captures aligned     │                                   │
│           │   front view           │                                   │
│           └────────────────────────┘                                   │
│                                                                         │
│  PARALLEL PROCESSING: While hardware moves to next attachment,         │
│                       Steps 3 & 4 analyze previous capture             │
│                                                                         │
│  OUTPUT: Front-view image for each attachment                          │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      STEP 3: FRONT VIEW DETECTION                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INPUT: Front-view capture                                              │
│  ┌─────────────────────────────────┐                                   │
│  │   [Front View of Attachment]    │                                   │
│  │                                 │                                   │
│  │         ┌──────────┐            │  ← AI detects attachment          │
│  │         │          │            │    in front view                  │
│  │         │   [●]    │            │                                   │
│  │         │          │            │                                   │
│  │         └──────────┘            │                                   │
│  └─────────────────────────────────┘                                   │
│                                                                         │
│  OUTPUT: Bounding box crop (with 50px padding)                         │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      STEP 4: DEFECT CLASSIFICATION                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INPUT: Bbox crop from Step 3                                           │
│  ┌──────────────────────┐                                              │
│  │                      │                                              │
│  │   ┌───────────┐      │  ← AI inspects for defects                  │
│  │   │   [✓]     │      │    (scratches, bent pins, etc.)             │
│  │   │  or [✗]   │      │                                              │
│  │   └───────────┘      │    Threshold: 0.5 (configurable)            │
│  └──────────────────────┘                                              │
│                                                                         │
│  OUTPUT: Defect annotations + confidence scores                        │
│  RESULT: PASS/FAIL per attachment                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### Inspection Workflow Explained (Technical Details)

#### Step 1: Top-View Detection
1. System captures image from top camera (or uses uploaded image)
2. Detectron2 runs inference with "Attachment" model
3. For each detection:
   - Extracts outer contour using edge detection
   - Computes outward-pointing arrow from center
   - Calculates phi angle (orientation in radians)
   - Assigns CCW index starting from bottom-right (315°)
4. **Outputs**:
   - `step-01_top_raw.png` - Original capture
   - `step-01_top_annotated.png` - With boxes, arrows, and indices
   - `step-01 cropped images/` - Individual detection crops

#### Step 2: Multi-Angle Capture
For each detection from Step 1 (in CCW order):
1. **Calculate motion targets**:
   - Turntable: Rotate by phi angle (degrees) for optimal front view
   - Linear axis: Position based on X-offset from center using FOV calibration
2. **Execute parallel motion** (turntable + axis move simultaneously)
3. **Capture sequence**:
   - Initial front capture
   - Crop based on calibrated FOV
   - (If axis available) Correction capture for verification
   - Final square crop at configured size
4. **Simultaneous processing**: While moving to next detection, Steps 3 & 4 run in background on previous capture
5. **Outputs**:
   - `step-02_front_initial_XXX.png` - Raw front view
   - `step-02_front_crop_initial_XXX.png` - FOV-cropped
   - `step-02_front_corrected_XXX.png` - After alignment verification
   - `step-02_front_XXX.png` - Final square crop
   - `step-02_top_XXX.png` - Top reference capture

#### Step 3: Front-View Detection
1. Runs "Front Attachment" model on each step-02 crop
2. Selects best detection (closest to center + highest score)
3. Extracts bounding box with 50px padding
4. **Outputs**:
   - `step-03_front_XXX.png` - Annotated with detection boxes
   - `step-03_front_bbox_XXX.png` - Padded bbox crop for defect analysis

#### Step 4: Defect Classification
1. Runs "Defect" model on each step-03 bbox crop
2. Applies configurable score threshold
3. Annotates all defects above threshold
4. **Outputs**:
   - `step-04_defect_XXX.png` - Annotated defect detections

### UI Component Guide

#### Camera Panel
- **Refresh Devices**: Re-enumerate connected cameras
- **Device Dropdown**: Shows all detected cameras (iRAYPLE + Basler combined)
- **Connect/Disconnect**: Green when connected
- **Capture Preview**: Takes test shot and displays in preview panel
- **Status Label**: Shows connected device name

#### Turntable Panel
- **Port Selection**: Lists available COM ports
- **Connect**: Establishes serial connection (115200 baud)
- **Home**: Returns to 0° (tracks offset from last position)
- **Step Angle**: Set rotation increment (-3600° to +3600°)
- **Rotate CCW/CW**: Manual rotation by step angle

#### Linear Axis Panel
- **Connect**: Serial connection at 9600 baud
- **Calibrate**: Auto-detects 0-100mm travel limits (LED turns green when complete)
- **Position**: Target position in mm (0-100)
- **Go**: Move to absolute position
- **Home**: Quick-return to configured home position
- **Set Home**: Save current position as new home

#### Light Controller
- **IP Address**: ULC-2 controller address (UDP port 9001)
- **Top Current**: Brightness for top camera (0-250 mA)
- **Front Current**: Brightness for front camera (0-250 mA)
- **Dwell Time**: Delay after brightness change (0-2000 ms)
- All settings auto-save on change

#### Edge/Contour Tuner
- **Purpose**: Fine-tune arrow/index computation parameters
- **Parameters**:
  - Blur kernel: Noise reduction before thresholding
  - Morphology: Clean up binary mask
  - Threshold offset: Adjust Otsu threshold
  - Approx epsilon: Polygon simplification tolerance
  - Smoothing: Chaikin curve iterations
  - Arrow length: Visual indicator size
- **Workflow**:
  1. Open Image or Capture Top
  2. Adjust parameters (live preview updates)
  3. Apply to Overlay (pushes to main preview)
  4. Close (saves parameters to state)

### Advanced Features

#### Re-running Detection on Existing Captures
1. Click **Run Step 3/4 on Existing Run**
2. Browse to previous capture directory (e.g., `captures/MyPart/2025-12-03/143022/`)
3. System re-scans `step-02/step_2_cropped/` for existing crops
4. Reruns Steps 3 & 4 with current model/threshold settings
5. **Use case**: Tune defect threshold without re-capturing

#### FOV Calibration
The system uses pre-calibrated field-of-view values for front camera alignment:
- `FRONT_FOV_TOP_PX = 951` - Width of front camera view in top image (pixels)
- `PIXELS_PER_MM = 72.3` - Conversion factor for linear axis
- **To recalibrate**:
  1. Place calibration target
  2. Measure front camera FOV in top image
  3. Update `state.front_fov_top_px` in code or config

#### Parallel Processing
- Steps 3 & 4 run in background ThreadPoolExecutor
- Overlaps with Step 2 motion for next detection
- Single worker ensures thread-safe Detectron2 access

---

## Configuration

### Persistent Settings (`user_settings.json`)
Located in application root directory. Auto-created on first run.

**Model Paths**:
```json
{
  "attachment_path": "C:/models/top_model.pth",
  "front_attachment_path": "C:/models/front_model.pth",
  "defect_path": "C:/models/defect_model.pth"
}
```

**Hardware Config**:
```json
{
  "camera_top_index": 0,
  "camera_front_index": 1,
  "turntable_port": "COM3",
  "turntable_step": 45,
  "linear_axis_port": "COM4",
  "linear_axis_home_mm": 50.0
}
```

**Detection Parameters**:
```json
{
  "solvision_score_threshold": null,  // Uses model metadata
  "defect_score_threshold": 0.5,
  "step2_crop_size": 1600,
  "front_fov_top_px": 951
}
```

**Light Settings**:
```json
{
  "light_ip": "192.168.1.100",
  "light_enabled": true,
  "top_current_ma": 200,
  "front_current_ma": 180,
  "light_dwell_ms": 60
}
```

### Model Metadata (`model_final.json`)
Each Detectron2 checkpoint should have an accompanying metadata file in the same directory:

```json
{
  "LearningParameter": {
    "class_names": "attachment",
    "class_colors": "#FF0000",
    "test_score_thresh": 0.8,
    "max_detections": 100,
    "min_dimension": 2592,
    "max_dimension": 3888
  },
  "ClassItems": [
    {"ID": 0, "Name": "attachment"}
  ]
}
```

**Required fields**:
- `class_names`: Comma-separated class names or via ClassItems
- `class_colors`: Comma-separated hex colors (one per class)
- `test_score_thresh`: Default inference threshold (optional)

---

## Data Management

### Capture Directory Structure
```
captures/
├── PartA/
│   ├── 2025-12-03/
│   │   ├── 093045/               # 09:30:45 capture
│   │   │   ├── step-01_top_raw.png
│   │   │   ├── step-01_top_annotated.png
│   │   │   ├── step-01 cropped images/
│   │   │   │   ├── crop_001.png
│   │   │   │   └── crop_002.png
│   │   │   ├── step-02/
│   │   │   │   ├── step-02_front_initial_001.png
│   │   │   │   ├── step-02_front_crop_initial_001.png
│   │   │   │   ├── step-02_front_corrected_001.png
│   │   │   │   ├── step-02_front_001.png
│   │   │   │   ├── step-02_top_001.png
│   │   │   │   └── step_2_cropped/
│   │   │   │       ├── step-02_front_crop_001.png
│   │   │   │       └── step-02_front_crop_002.png
│   │   │   ├── step-03/
│   │   │   │   ├── step-03_front_001.png
│   │   │   │   ├── step-03_front_bbox_001.png
│   │   │   │   └── ...
│   │   │   ├── step-04/
│   │   │   │   ├── step-04_defect_001.png
│   │   │   │   └── ...
│   │   │   └── cycle_time.txt    # Total elapsed time
│   │   └── 143022/               # 14:30:22 capture
│   └── 2025-12-04/
└── PartB/
```

### Data Extraction Tool
Flattens hierarchical structure for training data preparation:

```bash
python data_extractor.py
```

**Output**: `captures_extracted/` directory with flat naming:
```
captures_extracted/
├── PartA_2025-12-03_093045_step-01_top_raw.png
├── PartA_2025-12-03_093045_step-02_top_001.png
└── ...
```

**Collision handling**: Duplicates get `_2`, `_3` suffix

### Cycle Time Logs
Each run saves `cycle_time.txt`:
```
Total cycle time: 45.32 seconds
```

Useful for:
- Performance benchmarking
- Process optimization
- Hardware timing analysis

---

## Building Windows Executable

### Using the Build Script

The project includes a PowerShell script for creating standalone Windows executables with PyInstaller.

#### Basic Build
```powershell
# From project root (Windows PowerShell)
.\scripts\build_exe.ps1 -Python python -OutputName EnvistaSolomon
```

**Outputs**:
- `dist/EnvistaSolomon/EnvistaSolomon.exe` - Standalone executable
- `dist/EnvistaSolomon.zip` - Packaged distribution

#### Build Configuration
The script pins to CUDA 12.6 stack by default:
- `torch==2.7.1`
- `torchvision==0.22.1`
- `torchaudio==2.7.1`
- `detectron2==0.6+18f6958pt2.7.1cu126`

**Custom CUDA version** (advanced):
```powershell
.\scripts\build_exe.ps1 `
  -Python python `
  -OutputName EnvistaSolomon `
  -TorchIndex "https://download.pytorch.org/whl/cu118" `
  -TorchVersion "2.0.0" `
  -DetectronWheel "detectron2-0.6-cp39-cp39-win_amd64.whl"
```

#### Distribution Notes
- User data (`user_settings.json`, `captures/`, `crash.log`) writes next to `.exe`
- No installation required - unzip and run
- Requires CUDA 12.6 runtime on target machine
- Camera SDKs must be installed separately on target

### GitHub Actions CI/CD

Automated release builds via GitHub Actions workflow:

```bash
# Trigger from GitHub UI: Actions → build-release-exe
# Or via gh CLI:
gh workflow run build-release-exe.yml -f tag=v1.0.0
```

**Workflow inputs**:
- `tag` (required): Release version (e.g., `v1.0.0`)
- `torch_index_url`: PyTorch index (default: CUDA 12.6)
- `detectron_wheel_url`: Detectron2 wheel URL
- `python_version`: Python version (default: 3.12)

**Manual release** (after local build):
```bash
git tag v1.0.0
git push origin v1.0.0
gh release create v1.0.0 dist/EnvistaSolomon.zip \
  -t "Envista-Solomon v1.0.0" \
  -n "Release notes here"
```

---

## Troubleshooting

### Camera Issues

#### "No cameras detected"
**Causes**:
- Camera SDK not installed
- USB connection issue
- Insufficient permissions

**Solutions**:
1. Verify SDK installation:
   - **iRAYPLE**: Check `C:\Program Files\HuarayTech\MV Viewer\`
   - **Basler**: Run `pip show pypylon`
2. Check USB connection (try different port/cable)
3. Run MV Viewer or Pylon Viewer to test camera independently
4. Check Device Manager for driver status

#### "Camera connection failed" / "Timeout on grab"
**Causes**:
- Camera in use by another application
- Insufficient USB bandwidth
- Driver conflict

**Solutions**:
1. Close other camera applications (MV Viewer, Pylon Viewer)
2. Connect cameras to separate USB controllers
3. Reduce camera resolution if using USB 2.0
4. Reconnect camera (Disconnect → wait 5s → Connect)

#### "Frame conversion failed"
**Solutions**:
1. Update camera firmware via vendor tool
2. Check pixel format support (Mono8, BGR8)
3. Restart application

### Detection Issues

#### "Model failed to load"
**Causes**:
- Missing `model_final.json` metadata file
- Incompatible Detectron2 version
- Corrupted checkpoint file

**Solutions**:
1. Ensure `model_final.json` exists in same directory as `.pth` file
2. Verify metadata has `class_names` and `class_colors` fields
3. Re-export model from training environment
4. Check `crash.log` for detailed error

#### "CUDA out of memory"
**Solutions**:
1. Reduce `max_detections` in model metadata
2. Lower `max_dimension` in metadata (image resize)
3. Close other GPU applications
4. Fallback to CPU: Uninstall CUDA toolkit (app auto-detects)

#### "No detections found"
**Causes**:
- Score threshold too high
- Model not trained on similar data
- Poor lighting/focus

**Solutions**:
1. Lower threshold in workflow tab (try 0.3-0.5)
2. Check model was trained on similar parts
3. Adjust light controller settings
4. Use Edge Tuner to verify contour detection
5. Check camera focus and exposure

### Hardware Control Issues

#### Turntable "Connection failed"
**Solutions**:
1. Check COM port is correct (Device Manager → Ports)
2. Ensure no other app is using the port
3. Verify baud rate 115200 (check turntable docs)
4. Try different USB port
5. Install CH340/FTDI drivers

#### Turntable "Home failed" / "Rotation timeout"
**Solutions**:
1. Check power supply to turntable
2. Verify mechanical clearance (no obstructions)
3. Re-home after power cycle
4. Check serial cable quality

#### Linear Axis "Calibration failed"
**Solutions**:
1. Ensure axis is powered and initialized
2. Check mechanical limits (no binding)
3. Verify 9600 baud rate
4. Manually test with Arduino Serial Monitor
5. Re-flash Arduino firmware if available

#### Light Controller "Connection timeout"
**Solutions**:
1. Verify IP address (default: `192.168.1.100`)
2. Check network connectivity: `ping 192.168.1.100`
3. Ensure UDP port 9001 not blocked by firewall
4. Confirm ULC-2 is powered and on same network
5. Try static IP assignment on PC network adapter

### Performance Issues

#### "Detection very slow"
**Solutions**:
1. Verify GPU is being used:
   - Check log for `[Detectron] device: cuda`
   - Run `nvidia-smi` to see GPU activity
2. Reduce `max_dimension` in model metadata
3. Close other GPU applications (browsers, games)
4. Update NVIDIA drivers

#### "Turntable movement jerky"
**Solutions**:
1. Check power supply current rating
2. Reduce step angle (try 30° instead of 45°)
3. Add delays in turntable service if needed

### Application Crashes

#### "Silent exit" / "Unexpected crash"
**Solutions**:
1. Check `crash.log` in application directory
2. Common causes:
   - CUDA driver mismatch (reinstall CUDA 12.6)
   - Corrupt model file (re-download)
   - Out of memory (close other apps)
3. Run with Python directly (not .exe) for detailed traceback

#### "Qt platform plugin not found"
**Solutions**:
1. Reinstall PyQt5: `pip uninstall PyQt5 && pip install PyQt5`
2. Check `QT_PLUGIN_PATH` environment variable (should be unset)
3. For .exe: Verify `platforms/` folder in dist directory

### Data/Config Issues

#### "Settings not saving"
**Solutions**:
1. Check file permissions on application directory
2. Verify `user_settings.json` is not read-only
3. Run application as administrator (if in Program Files)

#### "Captures folder empty" / "Images not saving"
**Solutions**:
1. Check disk space
2. Verify Part ID is set (not empty)
3. Check file permissions on `captures/` folder
4. Look for errors in log panel

---

## Project Structure

```
Envista-Solomon/
├── main.py                      # Application entry point
├── requirements.txt             # Python dependencies
├── user_settings.json          # Persistent configuration (auto-generated)
├── crash.log                   # Error log (auto-generated)
│
├── services/                   # Backend services
│   ├── __init__.py
│   ├── app_paths.py           # Path utilities
│   ├── camera_backend_irayple.py  # iRAYPLE camera driver
│   ├── camera_backend_pylon.py    # Basler Pylon driver
│   ├── camera_manager.py      # Unified camera + light interface
│   ├── camera_service.py      # Camera enumeration & capture
│   ├── config.py              # Settings dataclasses
│   ├── contour_tools.py       # Edge detection & arrow computation
│   ├── crash_reporter.py      # Exception logging
│   ├── light_controller.py    # ULC-2 LED controller
│   ├── linear_axis_service.py # Linear stage control
│   ├── project_loader.py      # Model loading wrapper
│   ├── solvision_manager.py   # Detectron2 inference engine
│   └── turntable_service.py   # Rotary stage control
│
├── ui/                         # PyQt5 UI components
│   ├── __init__.py
│   ├── camera_panel.py        # Camera selection widget
│   ├── defect_ledger.py       # Defect results table
│   ├── edge_tuner.py          # Contour parameter tuning dialog
│   ├── image_preview_panel.py # Dual-pane image viewer
│   ├── init_wizard.py         # Setup wizard
│   ├── linear_axis_panel.py   # Linear axis controls
│   ├── loading_dialog.py      # Progress dialog
│   ├── logic_tab.py           # Logic tab (placeholder)
│   ├── main_window.py         # Main application window
│   ├── qt_image.py            # NumPy/Qt image conversion
│   ├── turntable_panel.py     # Turntable controls
│   └── workflow_tab.py        # Primary workflow interface
│
├── scripts/                    # Build & utility scripts
│   └── build_exe.ps1          # PyInstaller build script
│
├── data_extractor.py           # Capture flattening utility
├── live_blob_tool.py           # Interactive blob detection tuner
├── tmp_import_check.py         # Dependency validation
│
├── captures/                   # Output directory (auto-generated)
│   └── <PartID>/
│       └── <YYYY-MM-DD>/
│           └── <HHMMSS>/
│               ├── step-01_*.png
│               ├── step-02/
│               ├── step-03/
│               ├── step-04/
│               └── cycle_time.txt
│
├── captures_extracted/         # Flattened outputs (from data_extractor.py)
│
├── .github/                    # GitHub Actions workflows
│   └── workflows/
│       └── build-release-exe.yml
│
└── README.md                   # This file
```

### Key Files

| File | Purpose |
|------|---------|
| `main.py` | Bootstraps Detectron2, installs crash reporter, launches Qt app |
| `services/solvision_manager.py` | Detectron2 model loading, inference, result normalization |
| `services/camera_manager.py` | High-level camera + lighting control |
| `services/contour_tools.py` | Arrow computation, CCW indexing, edge detection |
| `ui/main_window.py` | 4-step detection orchestration, hardware coordination |
| `ui/workflow_tab.py` | User controls, settings, log display |
| `ui/image_preview_panel.py` | Overlay rendering (arrows, boxes, contours) |
| `data_extractor.py` | Training data preparation |

---

## Utilities

### Live Blob Tool
Interactive parameter tuning for blob detection (development/calibration):

```bash
# From webcam
python live_blob_tool.py

# From image file
python live_blob_tool.py --image path/to/test_image.jpg

# From video file
python live_blob_tool.py --video path/to/video.mp4

# From specific camera
python live_blob_tool.py --source 1
```

**Controls**:
- Trackbars adjust threshold, morphology, blob detector params
- Press `s` to save current frame
- Press `ESC` to quit

### Import Checker
Validates all dependencies are installed:

```bash
python tmp_import_check.py
```

Reports success/failure for each required package.

---

## Contributing

### Development Setup
1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Install dev dependencies: `pip install pytest black flake8`
4. Make changes and add tests
5. Run linter: `black . && flake8`
6. Commit: `git commit -m "Add your feature"`
7. Push: `git push origin feature/your-feature`
8. Open Pull Request

### Code Style
- **Formatting**: Black (line length 120)
- **Linting**: Flake8
- **Docstrings**: Google style
- **Type hints**: Encouraged for public APIs

### Testing
- Add tests for new features in `tests/` (if test suite exists)
- Ensure hardware mocking for CI compatibility
- Manual testing with real hardware required

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Detectron2**: Facebook AI Research for the object detection framework
- **PyQt5**: Riverbank Computing for the GUI toolkit
- **OpenCV**: Open Source Computer Vision Library
- **MindVision/iRAYPLE**: Industrial camera SDK
- **Basler**: Pylon camera SDK

---

## Support

For issues, questions, or feature requests:
1. Check [Troubleshooting](#troubleshooting) section
2. Search existing [GitHub Issues](https://github.com/yourusername/Envista-Solomon/issues)
3. Open a new issue with:
   - System info (OS, Python version, GPU)
   - Steps to reproduce
   - Relevant logs from `crash.log`
   - Screenshots if applicable

---

**Made with ❤️ for automated quality inspection**
