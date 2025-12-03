# Screenshots Directory

This directory should contain screenshots of the Envista-Solomon application for the main README.

## Required Screenshots

To complete the README documentation, please add the following screenshots:

### 1. `main_window.png`
- **What to capture**: Full application window
- **How to capture**:
  1. Run `python main.py`
  2. Complete initialization wizard
  3. Load all three models
  4. Connect cameras and hardware
  5. Take screenshot showing:
     - Workflow tab on left with all controls
     - Image preview panel on right with both panes
     - Detection table with sample results
     - Log panel with activity

### 2. `init_wizard.png`
- **What to capture**: Initialization wizard dialog
- **How to capture**:
  1. Run `python main.py` (first time or after clearing settings)
  2. Screenshot the wizard showing one of the setup steps
  3. Ideally show Step 2 (camera configuration) or Step 3 (preview)

### 3. `detection_results.png`
- **What to capture**: Top-view image with AI detection overlays
- **How to capture**:
  1. After running detection, screenshot the left pane (Attachment Overview)
  2. Should show:
     - Green bounding boxes around detections
     - Blue arrows pointing outward
     - Yellow index numbers (1, 2, 3...)
     - Red center crosshair

### 4. `defect_detection.png`
- **What to capture**: Defect analysis results
- **How to capture**:
  1. After Step 4 completes, screenshot showing:
     - Front view with defect annotations (if defects found)
     - Defect Ledger table with entries
     - Or step-04 output image with bounding boxes

### 5. `edge_tuner.png`
- **What to capture**: Edge/Contour Tuner dialog
- **How to capture**:
  1. Click "Edge/Contour Tuner" button in Workflow tab
  2. Load an image or capture from top camera
  3. Screenshot showing:
     - Parameter controls on left
     - Preview image on right with detected contours
     - Tuning sliders visible

## Screenshot Guidelines

- **Resolution**: 1920x1080 or higher
- **Format**: PNG (for lossless quality)
- **Cropping**: Crop out unnecessary desktop elements, focus on application
- **Privacy**: Ensure no sensitive part data or company information is visible
- **Quality**: Clear, well-lit, professional appearance

## Alternative: Add Demo Mode

If you cannot capture real hardware screenshots, consider adding:
- Sample images in the repository
- Demo/simulation mode that runs without hardware
- Pre-recorded results from test runs

## Usage in README

These images are referenced in the README via:
```markdown
![Main Window](docs/images/main_window.png)
```

Once added, they will automatically display in the GitHub README.
