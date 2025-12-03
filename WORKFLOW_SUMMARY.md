# Envista-Solomon Workflow Summary

Quick reference guide for the 4-step automated inspection workflow.

## Quick Overview

```
Part → Top Camera → AI Detection → Turntable Rotation → Front Captures → Defect Analysis → Results
  │         │            │               │                    │                │              │
  └─────────┴────────────┴───────────────┴────────────────────┴────────────────┴──────────────┘
                                    ~20-30 seconds total
```

## The 4 Steps

### Step 1: Top View Detection (3-5 sec)
- **Input**: Overhead camera image
- **Process**: AI detects all attachments, computes angles
- **Output**: Annotated image, detection list with coordinates & angles

### Step 2: Multi-Angle Capture (10-15 sec)
- **Input**: Detection list from Step 1
- **Process**: For each attachment:
  - Rotate turntable to optimal angle
  - Position linear axis to center
  - Capture front view
  - *Parallel*: Analyze previous capture while moving to next
- **Output**: Front-view images for each attachment

### Step 3: Front View Detection (2-5 sec)
- **Input**: Front-view images
- **Process**: AI detects attachment in each image, extracts bbox
- **Output**: Bounding box crops

### Step 4: Defect Classification (2-5 sec)
- **Input**: Bbox crops
- **Process**: AI inspects for defects with threshold
- **Output**: Pass/Fail per attachment with confidence scores

## Key Performance Features

1. **Parallel Processing**: Steps 3 & 4 run while Step 2 hardware moves
2. **Optimized Motion**: CCW ordering minimizes rotation angles
3. **FOV Calibration**: Precise linear axis positioning
4. **GPU Acceleration**: Fast AI inference when CUDA available

## Typical Cycle Times

| Scenario | Detections | Time | Notes |
|----------|-----------|------|-------|
| Small part | 2-3 | 15-20s | Minimal rotation |
| Medium part | 4-6 | 20-30s | Standard use case |
| Complex part | 8-10 | 35-50s | Many attachments |

*Times assume GPU available and calibrated hardware*

## Output Organization

All results saved to: `captures/<PartID>/<Date>/<Time>/`

- **step-01/**: Top view raw + annotated + crops
- **step-02/**: Front captures (initial, corrected, cropped)
- **step-03/**: Front detection annotated + bbox crops
- **step-04/**: Defect annotated images
- **cycle_time.txt**: Total elapsed time

## Configuration Options

- **Defect Threshold**: 0.0-1.0 (default: 0.5) - Lower = more sensitive
- **Crop Size**: 100-8192px (default: 1600) - Larger = more detail
- **Light Brightness**: 0-250mA per camera - Affects image quality
- **FOV Calibration**: 951px (default) - Affects linear axis accuracy

## Common Workflows

### Standard Inspection
```bash
1. Load three models (top, front, defect)
2. Enter Part ID
3. Click "Run Detection"
4. Wait for completion
5. Review results in preview panels and defect ledger
```

### Re-analyze Existing Captures
```bash
1. Adjust defect threshold
2. Click "Run Step 3/4 on Existing Run"
3. Browse to previous capture directory
4. System re-analyzes with new threshold
```

### Parameter Tuning
```bash
1. Click "Edge/Contour Tuner"
2. Load test image or capture from camera
3. Adjust parameters (blur, threshold, etc.)
4. Preview contours in real-time
5. Apply to overlay when satisfied
```

## Hardware Coordination

The system orchestrates 5 hardware components:
1. **Top Camera** - Overhead imaging
2. **Front Camera** - Detail inspection
3. **Turntable** - Part rotation (0-360°)
4. **Linear Axis** - Horizontal positioning (0-100mm)
5. **LED Controller** - Programmable lighting

All motion is **non-blocking** - UI remains responsive during inspection.

## AI Models

Three Detectron2 models work together:
1. **Attachment Model** (top) - Detects objects from overhead
2. **Front Model** - Detects objects in front views
3. **Defect Model** - Classifies quality (pass/fail)

Models are **hot-swappable** - load different checkpoints without restart.

## For More Details

See the main [README.md](README.md) for:
- Complete technical documentation
- Installation instructions
- Troubleshooting guide
- Hardware requirements
- Configuration options
