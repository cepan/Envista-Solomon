# User Acceptance Test (UAT) Document
## Envista-Solomon Inspection Machine

**Document Version**: 1.0
**Application Version**: Based on current codebase
**Date**: December 3, 2025
**Purpose**: User Acceptance Testing for automated multi-angle inspection system

---

## Table of Contents
1. [Introduction](#1-introduction)
2. [Test Environment Setup](#2-test-environment-setup)
3. [Test Scope](#3-test-scope)
4. [UAT Test Cases](#4-uat-test-cases)
5. [Pass/Fail Criteria](#5-passfail-criteria)
6. [Sign-off](#6-sign-off)

---

## 1. Introduction

### 1.1 Purpose
This document defines the User Acceptance Test criteria for the Envista-Solomon automated inspection system. The tests validate that the system meets all functional and operational requirements for production use.

### 1.2 System Overview
The Envista-Solomon system performs automated visual inspection of manufactured parts using:
- Dual industrial cameras (top and front views)
- Robotic positioning (turntable and linear axis)
- AI-powered defect detection (Detectron2)
- LED light control
- Automated result logging

### 1.3 Test Objectives
- Verify all UI components function as specified
- Confirm hardware integration works correctly
- Validate inspection workflow produces accurate results
- Ensure data traceability and storage
- Test error handling and system recovery

---

## 2. Test Environment Setup

### 2.1 Hardware Requirements
- [ ] Windows PC with NVIDIA GPU (CUDA 12.6)
- [ ] Top camera (iRAYPLE or Basler) connected
- [ ] Front camera (iRAYPLE or Basler) connected
- [ ] Turntable connected via serial (COM port)
- [ ] Linear axis connected via serial (COM port)
- [ ] ULC-2 LED controller on network
- [ ] Test part/sample available

### 2.2 Software Requirements
- [ ] Envista-Solomon application installed
- [ ] Python environment with all dependencies
- [ ] Detectron2 models available:
  - Top attachment model (.pth)
  - Front attachment model (.pth)
  - Defect detection model (.pth)
- [ ] Camera SDKs installed (iRAYPLE MV Viewer or pypylon)

### 2.3 Pre-Test Configuration
- [ ] `user_settings.json` cleared (for fresh initialization)
- [ ] `captures/` directory accessible with write permissions
- [ ] Network connectivity to LED controller verified
- [ ] Serial ports identified in Device Manager

---

## 3. Test Scope

### 3.1 In-Scope Features
✅ Initialization wizard
✅ Camera control and connection
✅ Motion control (turntable and linear axis)
✅ Light controller integration
✅ Model loading and management
✅ System health checks
✅ Inspection workflow (4-step pipeline)
✅ Result visualization and overlays
✅ Data storage and traceability
✅ Error handling and recovery

### 3.2 Out-of-Scope
❌ Model training or re-training
❌ Camera firmware updates
❌ Hardware calibration (assumes pre-calibrated)
❌ Network infrastructure setup

---

## 4. UAT Test Cases

---

## TEST SECTION 1: INITIALIZATION VIEW (SYSTEM SETUP)

### TC-INIT-001: Launch Initialization Wizard
**Objective**: Verify initialization wizard appears on first run
**Preconditions**: `user_settings.json` deleted or does not exist
**Steps**:
1. Launch application: `python main.py`
2. Observe initialization wizard appears

**Expected Results**:
- [x] Initialization wizard displays with 4 steps
- [x] Step 1: Model Selection visible
- [x] Step 2: Camera Configuration visible
- [x] Step 3: Live Preview visible
- [x] Step 4: Turntable Setup visible
- [x] "Begin Workflow" button visible but disabled

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-CAM-001: Camera Enumeration (Top Camera)
**Objective**: Verify top camera can be detected and listed
**Preconditions**: Top camera connected via USB
**Steps**:
1. In Init Wizard Step 2, locate "Top Camera" section
2. Click "Refresh Devices" button
3. Observe device dropdown

**Expected Results**:
- [x] Camera device appears in "Top Camera" dropdown
- [x] Device name shows camera model or serial number
- [x] Backend type indicated (iRAYPLE or Pylon)
- [x] Status shows "Disconnected"

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-CAM-002: Camera Connection (Top Camera)
**Objective**: Verify top camera can connect successfully
**Preconditions**: Camera enumerated (TC-CAM-001 passed)
**Steps**:
1. Select top camera from dropdown
2. Click "Connect" button
3. Wait for connection to complete

**Expected Results**:
- [x] Connect button changes to green
- [x] Button text changes to "Disconnect"
- [x] Status label shows "Connected: [device name]"
- [x] Capture Preview button becomes enabled
- [x] No error messages in log

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-CAM-003: Camera Live Preview (Top Camera)
**Objective**: Verify top camera can capture and display preview
**Preconditions**: Top camera connected (TC-CAM-002 passed)
**Steps**:
1. Click "Capture Preview" button
2. Navigate to Step 3 (Live Preview) in wizard
3. Observe left pane (Attachment Overview)

**Expected Results**:
- [x] Image appears in preview pane
- [x] Image shows current view from top camera
- [x] Image scaled to fit pane
- [x] No corruption or artifacts

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-CAM-004: Camera Disconnection (Top Camera)
**Objective**: Verify top camera can disconnect cleanly
**Preconditions**: Top camera connected (TC-CAM-002 passed)
**Steps**:
1. Click "Disconnect" button
2. Observe status change

**Expected Results**:
- [x] Button changes from green to default color
- [x] Button text changes to "Connect"
- [x] Status label shows "Disconnected"
- [x] Capture Preview button becomes disabled
- [x] No error messages

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-CAM-005: Camera Enumeration (Front Camera)
**Objective**: Verify front camera can be detected separately
**Preconditions**: Front camera connected via USB
**Steps**:
1. In Camera Panel, locate "Front Camera" section
2. Click "Refresh Devices" button
3. Observe device dropdown

**Expected Results**:
- [x] Front camera appears in dropdown (different from top)
- [x] Device name shows camera model or serial
- [x] Backend type indicated
- [x] Can select different camera than top

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-CAM-006: Camera Connection (Front Camera)
**Objective**: Verify front camera can connect while top connected
**Preconditions**: Top camera connected, front enumerated
**Steps**:
1. Select front camera from dropdown
2. Click "Connect" button
3. Verify both cameras connected

**Expected Results**:
- [x] Front camera connects successfully
- [x] Both cameras show "Connected" status
- [x] Both Capture Preview buttons enabled
- [x] No resource conflicts

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-CAM-007: Camera Error - Same Device
**Objective**: Verify system prevents assigning same camera twice
**Preconditions**: One camera connected
**Steps**:
1. Connect top camera to device #0
2. Attempt to connect front camera to same device #0

**Expected Results**:
- [x] System prevents connection OR shows warning
- [x] Error message indicates device already in use
- [x] Previous connection maintained

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-CAM-008: Camera Error - Capture Timeout
**Objective**: Verify error handling when camera times out
**Preconditions**: Camera connected but blocked/covered
**Steps**:
1. Cover camera lens or disconnect after connection
2. Click "Capture Preview"
3. Wait for timeout

**Expected Results**:
- [x] Error message displayed
- [x] Log shows "timeout on grab" or similar
- [x] Application remains responsive
- [x] Can retry capture after recovery

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

## TEST SECTION 2: MOTION CONTROL

### TC-TT-001: Turntable Enumeration
**Objective**: Verify turntable serial port can be detected
**Preconditions**: Turntable connected via serial adapter
**Steps**:
1. In Turntable Panel, click "Refresh" button
2. Observe port dropdown

**Expected Results**:
- [x] COM port appears in dropdown (e.g., COM3)
- [x] Multiple ports shown if multiple adapters present
- [x] Status shows "Disconnected"

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-TT-002: Turntable Connection
**Objective**: Verify turntable can connect successfully
**Preconditions**: COM port detected (TC-TT-001 passed)
**Steps**:
1. Select turntable port from dropdown
2. Click "Connect" button
3. Wait for connection to complete

**Expected Results**:
- [x] Connect button changes to green
- [x] Button text changes to "Disconnect"
- [x] Status message shows connection success
- [x] Motion controls become enabled
- [x] Baud rate 115200 used

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-TT-003: Turntable Homing
**Objective**: Verify turntable can home to 0° position
**Preconditions**: Turntable connected (TC-TT-002 passed)
**Steps**:
1. Click "Home" button
2. Observe turntable motion
3. Wait for completion

**Expected Results**:
- [x] Turntable rotates to home position (0°)
- [x] Status message shows "Homed" or similar
- [x] Position feedback displays 0.0°
- [x] Motion state transitions: Idle → Moving → Idle
- [x] No mechanical binding or faults

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-TT-004: Turntable Jog Positive
**Objective**: Verify turntable can rotate clockwise
**Preconditions**: Turntable homed (TC-TT-003 passed)
**Steps**:
1. Set "Step Angle" to 45°
2. Click "Rotate CW" button
3. Observe rotation

**Expected Results**:
- [x] Turntable rotates 45° clockwise
- [x] Position feedback updates (e.g., 45.0°)
- [x] Motion completes smoothly
- [x] Can repeat multiple times

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-TT-005: Turntable Jog Negative
**Objective**: Verify turntable can rotate counter-clockwise
**Preconditions**: Turntable homed (TC-TT-003 passed)
**Steps**:
1. Set "Step Angle" to 45°
2. Click "Rotate CCW" button
3. Observe rotation

**Expected Results**:
- [x] Turntable rotates 45° counter-clockwise
- [x] Position feedback updates (e.g., -45.0° or 315.0°)
- [x] Motion completes smoothly
- [x] Negative angles handled correctly

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-TT-006: Turntable Error - Not Homed
**Objective**: Verify system detects un-homed state
**Preconditions**: Turntable connected but not homed
**Steps**:
1. Connect turntable
2. Do NOT click Home
3. Attempt to start inspection

**Expected Results**:
- [x] System health check shows "Turntable homed: NG"
- [x] START button remains disabled
- [x] Error message indicates homing required
- [x] Log shows warning

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-AXIS-001: Linear Axis Enumeration
**Objective**: Verify linear axis serial port can be detected
**Preconditions**: Linear axis connected via serial adapter
**Steps**:
1. In Linear Axis Panel, click "Refresh" button
2. Observe port dropdown

**Expected Results**:
- [x] COM port appears in dropdown
- [x] Port different from turntable if both connected
- [x] Status shows "Disconnected"

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-AXIS-002: Linear Axis Connection
**Objective**: Verify linear axis can connect successfully
**Preconditions**: COM port detected (TC-AXIS-001 passed)
**Steps**:
1. Select axis port from dropdown
2. Click "Connect" button
3. Wait for connection

**Expected Results**:
- [x] Connect button changes to green
- [x] Button text changes to "Disconnect"
- [x] Status shows connection success
- [x] Motion controls enabled
- [x] Baud rate 9600 used

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-AXIS-003: Linear Axis Calibration
**Objective**: Verify linear axis can auto-calibrate travel limits
**Preconditions**: Axis connected (TC-AXIS-002 passed)
**Steps**:
1. Click "Calibrate" button
2. Observe axis motion (left sweep, then right sweep)
3. Wait for completion

**Expected Results**:
- [x] Button turns red during calibration
- [x] Axis moves to left limit
- [x] Axis moves to right limit (100mm)
- [x] Button turns green when complete
- [x] Status shows "Calibrated"
- [x] Position feedback shows current position

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-AXIS-004: Linear Axis Position Control
**Objective**: Verify axis can move to absolute position
**Preconditions**: Axis calibrated (TC-AXIS-003 passed)
**Steps**:
1. Set "Position" spinner to 50.0 mm
2. Click "Go" button
3. Observe motion

**Expected Results**:
- [x] Axis moves to 50mm position
- [x] Position feedback updates (±0.1mm accuracy)
- [x] Motion state: Idle → Moving → Idle
- [x] No overshooting or oscillation

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-AXIS-005: Linear Axis Homing
**Objective**: Verify axis can return to home position
**Preconditions**: Axis calibrated and moved (TC-AXIS-004 passed)
**Steps**:
1. Move axis to 80mm
2. Click "Home" button
3. Observe motion

**Expected Results**:
- [x] Axis moves to default home (50mm)
- [x] Position feedback shows home position
- [x] Motion completes smoothly

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-AXIS-006: Linear Axis Custom Home
**Objective**: Verify custom home position can be set
**Preconditions**: Axis calibrated
**Steps**:
1. Move axis to 30mm using Go button
2. Click "Set Home to Position" button
3. Move axis to 80mm
4. Click "Home" button

**Expected Results**:
- [x] Custom home (30mm) saved
- [x] Home button returns to 30mm (not default 50mm)
- [x] Setting persists across sessions

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-AXIS-007: Linear Axis Error - Not Calibrated
**Objective**: Verify system detects uncalibrated state
**Preconditions**: Axis connected but not calibrated
**Steps**:
1. Connect axis
2. Do NOT calibrate
3. Check system health

**Expected Results**:
- [x] System health shows "Linear axis homed: NG"
- [x] Position commands may be ignored or warned
- [x] Log shows calibration required

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

## TEST SECTION 3: LIGHT CONTROLLER

### TC-LIGHT-001: Light Controller Connection
**Objective**: Verify LED controller can be reached on network
**Preconditions**: ULC-2 controller powered and on network
**Steps**:
1. In Light Controller section, enter IP address (e.g., 192.168.1.100)
2. System attempts connection automatically
3. Observe status

**Expected Results**:
- [x] IP address field accepts input
- [x] Connection status shows "Online" or "Offline"
- [x] UDP port 9001 used
- [x] No network errors if reachable

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-LIGHT-002: Light Intensity - Top Camera
**Objective**: Verify top camera light intensity can be adjusted
**Preconditions**: Light controller online (TC-LIGHT-001 passed)
**Steps**:
1. Set "Top Current" slider to 100 mA
2. Observe light output
3. Adjust to 200 mA
4. Observe brightness change

**Expected Results**:
- [x] Light intensity changes corresponding to slider
- [x] Range: 0-250 mA
- [x] Setting saves automatically
- [x] Read-back verification confirms setting applied

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-LIGHT-003: Light Intensity - Front Camera
**Objective**: Verify front camera light can be controlled independently
**Preconditions**: Light controller online
**Steps**:
1. Set "Front Current" slider to 150 mA
2. Verify independent from top light
3. Adjust to 200 mA

**Expected Results**:
- [x] Front light adjusts independently
- [x] Top light setting unchanged
- [x] Both lights can be different intensities
- [x] Settings persist

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-LIGHT-004: Light Dwell Time
**Objective**: Verify dwell time setting affects capture delay
**Preconditions**: Light controller online
**Steps**:
1. Set "Dwell Time" to 100 ms
2. Change top light intensity
3. Trigger capture immediately
4. Observe delay before capture

**Expected Results**:
- [x] System waits 100ms after light change before capture
- [x] Ensures light stabilizes before imaging
- [x] Range: 0-2000 ms
- [x] Setting saves automatically

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-LIGHT-005: Light Error - Network Unreachable
**Objective**: Verify error handling when controller offline
**Preconditions**: Incorrect IP or controller powered off
**Steps**:
1. Enter invalid IP address (e.g., 192.168.1.254)
2. Attempt brightness change
3. Observe status

**Expected Results**:
- [x] Status shows "Offline"
- [x] System health shows "Light reachable: NG"
- [x] Error logged but system remains functional
- [x] Can continue without lights (warning mode)

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

## TEST SECTION 4: MODEL MANAGEMENT

### TC-MODEL-001: Model Loading - Top Attachment
**Objective**: Verify top attachment model can be loaded
**Preconditions**: Valid Detectron2 .pth file available
**Steps**:
1. In Workflow Tab, locate "Top Attachment Model" section
2. Click "Browse" button
3. Navigate to model file
4. Select `top_model.pth`
5. Click "Load" button

**Expected Results**:
- [x] File browser opens
- [x] Can navigate to model directory
- [x] Model file selected
- [x] Path displayed in text field
- [x] Load button changes to green when loaded
- [x] Status indicator shows "Loaded"
- [x] Log shows model loading success with class count

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-MODEL-002: Model Loading - Front Attachment
**Objective**: Verify front attachment model can be loaded independently
**Preconditions**: Valid front model .pth file available
**Steps**:
1. Locate "Front Attachment Model" section
2. Browse and select `front_model.pth`
3. Click Load

**Expected Results**:
- [x] Front model loads successfully
- [x] Independent from top model
- [x] Status shows "Loaded"
- [x] Green indicator appears

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-MODEL-003: Model Loading - Defect Detection
**Objective**: Verify defect detection model can be loaded
**Preconditions**: Valid defect model .pth file available
**Steps**:
1. Locate "Defect Detection Model" section
2. Browse and select `defect_model.pth`
3. Click Load

**Expected Results**:
- [x] Defect model loads successfully
- [x] Independent from other models
- [x] Status shows "Loaded"
- [x] All three models can be loaded simultaneously

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-MODEL-004: Model Metadata Parsing
**Objective**: Verify model metadata (model_final.json) is read correctly
**Preconditions**: Model with model_final.json in same directory
**Steps**:
1. Load model with metadata file
2. Observe log output
3. Check for class names, colors, threshold

**Expected Results**:
- [x] System reads model_final.json
- [x] Class names extracted (e.g., "attachment")
- [x] Class colors parsed (hex codes)
- [x] Score threshold loaded
- [x] Max detections setting applied
- [x] Image dimension settings applied

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-MODEL-005: Model Error - Missing Metadata
**Objective**: Verify error handling when model_final.json missing
**Preconditions**: Model .pth file without model_final.json
**Steps**:
1. Attempt to load model without metadata
2. Observe error

**Expected Results**:
- [x] Error message displayed
- [x] Log shows "Model metadata not found"
- [x] Model fails to load
- [x] Status shows "Error" or "Not loaded"
- [x] System remains stable

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-MODEL-006: Model Error - Missing Class Colors
**Objective**: Verify error when class_colors missing in metadata
**Preconditions**: model_final.json without class_colors field
**Steps**:
1. Load model with incomplete metadata
2. Observe error

**Expected Results**:
- [x] Error message: "class_colors not found"
- [x] Model fails to load
- [x] Clear indication of what's missing
- [x] Guidance to fix metadata file

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-MODEL-007: Model Reload
**Objective**: Verify models can be reloaded without restart
**Preconditions**: Model already loaded
**Steps**:
1. Update model file or metadata
2. Click "Load" button again (reload)
3. Verify new version loads

**Expected Results**:
- [x] Old model released from memory
- [x] New model loaded successfully
- [x] No application restart required
- [x] Updated class names/settings applied

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-MODEL-008: GPU vs CPU Detection
**Objective**: Verify system detects and uses available hardware
**Preconditions**: System with or without CUDA GPU
**Steps**:
1. Load any model
2. Check log output for device info

**Expected Results**:
- [x] Log shows "device: cuda" if GPU available
- [x] Log shows "device: cpu" if no GPU
- [x] Models load successfully on either
- [x] Inference runs (slower on CPU)

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

## TEST SECTION 5: SYSTEM HEALTH CHECK

### TC-HEALTH-001: Health Check - All Systems OK
**Objective**: Verify health check passes when all systems ready
**Preconditions**: All hardware connected, homed, and models loaded
**Steps**:
1. Complete all initialization steps
2. Observe initialization wizard validation
3. Check "Begin Workflow" button state

**Expected Results**:
- [x] Top camera: OK ✓
- [x] Front camera: OK ✓
- [x] Turntable homed: OK ✓
- [x] Linear axis homed: OK ✓
- [x] Light reachable: OK ✓
- [x] Top model loaded: OK ✓
- [x] Front model loaded: OK ✓
- [x] Defect model loaded: OK ✓
- [x] Storage path reachable: OK ✓
- [x] "Begin Workflow" button ENABLED

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-HEALTH-002: Health Check - Camera Failure
**Objective**: Verify health check blocks start when camera disconnected
**Preconditions**: One camera disconnected
**Steps**:
1. Disconnect top camera
2. Check health status
3. Attempt to proceed

**Expected Results**:
- [x] Top camera: NG ✗
- [x] Error message shows "Top camera not connected"
- [x] "Begin Workflow" button DISABLED
- [x] Cannot proceed to main window

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-HEALTH-003: Health Check - Motion Not Homed
**Objective**: Verify health check detects un-homed axes
**Preconditions**: Turntable connected but not homed
**Steps**:
1. Connect turntable but skip homing
2. Check health status

**Expected Results**:
- [x] Turntable homed: NG ✗
- [x] Error message shows "Turntable not homed"
- [x] "Begin Workflow" button DISABLED
- [x] Instruction to home turntable displayed

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-HEALTH-004: Health Check - Model Not Loaded
**Objective**: Verify health check detects missing models
**Preconditions**: At least one model not loaded
**Steps**:
1. Load top and front models
2. Do NOT load defect model
3. Check health status

**Expected Results**:
- [x] Defect model loaded: NG ✗
- [x] Error message shows which model missing
- [x] "Begin Workflow" button DISABLED
- [x] Instruction to load model displayed

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-HEALTH-005: Health Check - Storage Inaccessible
**Objective**: Verify health check detects storage issues
**Preconditions**: `captures/` directory missing or read-only
**Steps**:
1. Make captures directory read-only OR delete it
2. Check health status

**Expected Results**:
- [x] Storage path reachable: NG ✗
- [x] Error message shows storage issue
- [x] Cannot start inspection
- [x] Path displayed to user

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-HEALTH-006: Health Check Recovery
**Objective**: Verify health check updates when issues resolved
**Preconditions**: Health check failing (TC-HEALTH-002 to 005)
**Steps**:
1. Start with failing health check
2. Resolve the issue (connect camera, home axis, load model, etc.)
3. Observe health check update

**Expected Results**:
- [x] Health status updates in real-time
- [x] NG → OK transition occurs
- [x] "Begin Workflow" button enables when all OK
- [x] No application restart required

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

## TEST SECTION 6: MAIN INSPECTION VIEW

### TC-INSP-001: Inspection Enable Rule
**Objective**: Verify START button only enables when all checks pass
**Preconditions**: In main window after wizard completion
**Steps**:
1. Enter Part ID in workflow tab
2. Locate "Run Detection" button
3. Verify button state

**Expected Results**:
- [x] Button enabled if all health checks OK
- [x] Button disabled if any check NG
- [x] Tooltip or status shows reason if disabled

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-INSP-002: Job Context Display
**Objective**: Verify job information is displayed and tracked
**Preconditions**: Ready to start inspection
**Steps**:
1. Enter Part ID: "TestPart_001"
2. Start inspection
3. Observe job context

**Expected Results**:
- [x] Part ID displayed: "TestPart_001"
- [x] Timestamp shows current date/time
- [x] Auto-generated JobID format: YYYYMMDD/HHMMSS
- [x] Storage path shown: `captures/TestPart_001/YYYY-MM-DD/HHMMSS/`

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-INSP-003: Inspection Start - Full Workflow
**Objective**: Verify complete inspection workflow executes
**Preconditions**: All systems OK, Part ID entered
**Steps**:
1. Click "Run Detection" button
2. Observe workflow execution
3. Wait for completion

**Expected Results**:

**Step 1 - Top View Detection**:
- [x] Top camera captures image
- [x] AI model runs inference
- [x] Detections displayed in table with:
  - Index (#1, #2, etc.)
  - Class name
  - Confidence score
  - Angle (degrees)
  - Phi (radians)
  - Center coordinates
  - Bounding box
- [x] Preview pane shows annotated image with:
  - Green bounding boxes
  - Blue arrows pointing outward
  - Yellow index numbers
  - Red center crosshair
- [x] Log shows detection count

**Step 2 - Multi-Angle Capture**:
- [x] For each detection (CCW order):
  - Turntable rotates to phi angle
  - Linear axis moves to position (if available)
  - Front camera captures view
  - Log shows progress
- [x] Motion occurs smoothly without collisions
- [x] Parallel processing occurs (Steps 3 & 4 run in background)

**Step 3 - Front View Detection**:
- [x] Front attachment model runs on crops
- [x] Best detection selected per crop
- [x] Bounding boxes extracted

**Step 4 - Defect Classification**:
- [x] Defect model runs on bbox crops
- [x] Defects (if any) displayed in Defect Ledger
- [x] Confidence scores shown
- [x] Pass/Fail status per attachment

**Completion**:
- [x] Cycle time displayed (total seconds)
- [x] Overall result: OK or NG
- [x] All images saved to storage
- [x] Log shows "Inspection complete"

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-INSP-004: Visual Inspection Context
**Objective**: Verify live camera feeds and overlays work correctly
**Preconditions**: Inspection running
**Steps**:
1. Observe left preview pane (Attachment Overview)
2. Observe right preview pane (Front Inspection)
3. Toggle "Overlay" checkbox

**Expected Results**:
- [x] Left pane shows top camera image with 2x stretching
- [x] Right pane shows front camera crops
- [x] Overlays toggle on/off:
  - Detection boxes
  - Arrows
  - Indices
  - Contours
  - Center crosshair
- [x] Images scale to fit panes
- [x] No distortion or aspect ratio issues

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-INSP-005: Attachment Status Display
**Objective**: Verify attachment list updates during inspection
**Preconditions**: Inspection in progress
**Steps**:
1. Observe detection table during workflow
2. Monitor status changes per attachment

**Expected Results**:
- [x] Table populates after Step 1
- [x] Each attachment shows:
  - Index (1, 2, 3...)
  - Class name
  - Score
  - Angle
  - Phi
  - Center coordinates
  - Bounds
- [x] Can select row to view details
- [x] Table remains accessible after completion

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-INSP-006: Defect List Display
**Objective**: Verify defect list populates correctly
**Preconditions**: Inspection complete with defects detected
**Steps**:
1. Complete inspection with known defects
2. Locate Defect Ledger panel
3. Review defect entries

**Expected Results**:
- [x] Defect Ledger displays table with:
  - Index (attachment number)
  - Class (defect type)
  - Confidence score
  - Area (pixels)
  - Bounds (x, y, w, h)
- [x] Each defect linked to specific attachment
- [x] Multiple defects per attachment shown
- [x] Empty if no defects (PASS result)

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-INSP-007: Log Output
**Objective**: Verify real-time logging works correctly
**Preconditions**: Application running
**Steps**:
1. Locate log panel in workflow tab
2. Perform various actions (connect, home, detect)
3. Observe log entries

**Expected Results**:
- [x] Log displays timestamped entries
- [x] Severity levels visible:
  - [INFO] for normal operations
  - [WARNING] for non-critical issues
  - [ERROR] for failures
- [x] Log auto-scrolls to latest entry
- [x] Entries include:
  - Camera connections
  - Motion commands
  - Model loading
  - Detection progress
  - Completion status

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-INSP-008: System State Indicators
**Objective**: Verify system state is clearly communicated
**Preconditions**: Application running
**Steps**:
1. Observe system state during different phases:
   - Idle/Ready
   - Running inspection
   - Error condition

**Expected Results**:
- [x] State visible in UI (log or status bar)
- [x] States include:
  - READY: System idle, can start
  - RUNNING: Inspection in progress
  - ERROR: Fault occurred
- [x] State transitions clearly indicated
- [x] Cannot start new inspection while RUNNING

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-INSP-009: Stop/Abort Inspection
**Objective**: Verify inspection can be stopped mid-execution
**Preconditions**: Inspection running
**Steps**:
1. Start inspection
2. During Step 2 motion, press STOP (ESC key or close window)
3. Observe behavior

**Expected Results**:
- [x] Inspection stops gracefully
- [x] Hardware motion halts safely
- [x] Partial results saved
- [x] System returns to READY state
- [x] Can start new inspection after stop

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-INSP-010: Re-run Inspection (Step 3/4 Only)
**Objective**: Verify can re-analyze existing captures
**Preconditions**: Previous inspection completed
**Steps**:
1. Click "Run Step 3/4 on Existing Run" button
2. Browse to previous capture directory
3. Select folder (e.g., `captures/TestPart/2025-12-03/143022/`)
4. Adjust defect threshold if desired
5. Confirm

**Expected Results**:
- [x] File browser opens
- [x] Can navigate to captures directory
- [x] System re-scans `step-02/step_2_cropped/` folder
- [x] Steps 3 & 4 re-execute with current models/settings
- [x] New step-03 and step-04 outputs generated
- [x] Original step-01 and step-02 images unchanged
- [x] Useful for threshold tuning without re-capturing

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

## TEST SECTION 7: RESULTS & TRACEABILITY

### TC-RESULT-001: Image Storage Structure
**Objective**: Verify all images saved to correct locations
**Preconditions**: Inspection completed
**Steps**:
1. Navigate to `captures/` directory
2. Locate job folder: `<PartID>/<YYYY-MM-DD>/<HHMMSS>/`
3. Verify file structure

**Expected Results**:
- [x] Directory structure exists:
```
captures/TestPart_001/2025-12-03/143022/
├── step-01_top_raw.png
├── step-01_top_annotated.png
├── step-01 cropped images/
│   ├── crop_001.png
│   └── crop_002.png
├── step-02/
│   ├── step-02_front_initial_001.png
│   ├── step-02_front_crop_initial_001.png
│   ├── step-02_front_corrected_001.png
│   ├── step-02_front_001.png
│   ├── step-02_top_001.png
│   └── step_2_cropped/
│       ├── step-02_front_crop_001.png
│       └── step-02_front_crop_002.png
├── step-03/
│   ├── step-03_front_001.png
│   ├── step-03_front_bbox_001.png
│   └── ...
├── step-04/
│   ├── step-04_defect_001.png
│   └── ...
└── cycle_time.txt
```
- [x] All files present
- [x] Images readable (not corrupted)
- [x] Naming convention followed

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-RESULT-002: Cycle Time Recording
**Objective**: Verify cycle time logged correctly
**Preconditions**: Inspection completed
**Steps**:
1. Open `cycle_time.txt` in job folder
2. Verify contents

**Expected Results**:
- [x] File contains single line
- [x] Format: "Total cycle time: XX.XX seconds"
- [x] Time is reasonable (15-60 seconds typical)
- [x] Matches observed inspection duration

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-RESULT-003: Data Extraction Tool
**Objective**: Verify data extractor flattens captures correctly
**Preconditions**: Multiple inspection runs completed
**Steps**:
1. Run: `python data_extractor.py`
2. Wait for completion
3. Check `captures_extracted/` directory

**Expected Results**:
- [x] `captures_extracted/` directory created
- [x] Files have flattened naming:
  - `PartID_YYYY-MM-DD_HHMMSS_step-01_top_raw.png`
  - `PartID_YYYY-MM-DD_HHMMSS_step-02_top_001.png`
- [x] All step-01 and step-02 files copied
- [x] Collision handling: duplicates get `_2`, `_3` suffix
- [x] Original files unchanged

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-RESULT-004: Model Traceability
**Objective**: Verify model versions traceable per job
**Preconditions**: Inspection completed
**Steps**:
1. Check log output from inspection
2. Verify model info displayed
3. Note model paths used

**Expected Results**:
- [x] Log shows model loading messages:
  - "Loaded model for 'top': <path>"
  - "Loaded model for 'front': <path>"
  - "Loaded model for 'defect': <path>"
- [x] Class names and counts logged
- [x] Score thresholds logged
- [x] Model paths recorded in memory
- [x] Can query current models via diagnostics

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-RESULT-005: Configuration Persistence
**Objective**: Verify settings persist across sessions
**Preconditions**: Settings configured
**Steps**:
1. Configure all settings (camera indices, ports, light, models)
2. Close application
3. Reopen application
4. Verify settings loaded

**Expected Results**:
- [x] `user_settings.json` updated on changes
- [x] Settings reload on startup:
  - Last camera selections
  - Last COM ports
  - Last model paths
  - Light IP and brightness
  - Crop size
  - Defect threshold
  - Part ID
  - Turntable step angle
  - Linear axis home position
- [x] No re-configuration needed

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

## TEST SECTION 8: ERROR HANDLING

### TC-ERROR-001: Camera Disconnection During Inspection
**Objective**: Verify error handling when camera disconnects mid-run
**Preconditions**: Inspection running
**Steps**:
1. Start inspection
2. Physically disconnect camera USB during Step 2
3. Observe error handling

**Expected Results**:
- [x] Error message displayed
- [x] Log shows: "Camera timeout" or "grab failed"
- [x] Inspection stops gracefully
- [x] Partial results saved
- [x] System transitions to ERROR state
- [x] Can recover by reconnecting camera

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-ERROR-002: Motion Fault During Inspection
**Objective**: Verify error handling for motion faults
**Preconditions**: Inspection running
**Steps**:
1. Start inspection
2. Introduce mechanical obstruction to turntable/axis
3. Observe error handling

**Expected Results**:
- [x] Motion timeout detected
- [x] Error message shows: "Motion fault" or "timeout"
- [x] Hardware motion stops
- [x] Inspection aborts
- [x] Error state indicated
- [x] Log shows which axis faulted

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-ERROR-003: AI Model Inference Failure
**Objective**: Verify error handling when AI inference fails
**Preconditions**: Inspection running
**Steps**:
1. Start inspection
2. Simulate CUDA out of memory (run other GPU apps)
3. OR use corrupted image
4. Observe error handling

**Expected Results**:
- [x] Inference error caught
- [x] Error message displayed
- [x] Log shows: "CUDA out of memory" or "inference failed"
- [x] Inspection stops
- [x] System recovers to READY state
- [x] Can retry after resolving issue

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-ERROR-004: Storage Full or Inaccessible
**Objective**: Verify error handling when disk full or path removed
**Preconditions**: Inspection starting
**Steps**:
1. Fill disk to <1GB free OR make captures folder read-only
2. Attempt inspection
3. Observe error

**Expected Results**:
- [x] Error detected before or during save
- [x] Message: "Storage path inaccessible" or "disk full"
- [x] Inspection stops before data loss
- [x] Log shows storage error
- [x] User directed to resolve storage issue

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-ERROR-005: Crash Recovery
**Objective**: Verify crash logging captures unexpected errors
**Preconditions**: Simulate application crash
**Steps**:
1. Trigger unhandled exception (if possible)
2. Application crashes or exits
3. Check `crash.log` file

**Expected Results**:
- [x] `crash.log` created in app directory
- [x] Log contains:
  - Timestamp
  - Exception type
  - Full traceback
  - Thread info (if background thread)
- [x] User can report crash log for debugging
- [x] Application can restart cleanly

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-ERROR-006: Error Source Identification
**Objective**: Verify all errors clearly indicate source
**Preconditions**: Various error conditions
**Steps**:
1. Trigger different errors:
   - Camera error
   - Axis error
   - Model error
   - Storage error
2. Review error messages

**Expected Results**:
- [x] Each error includes source:
  - "Camera: <error>"
  - "Turntable: <error>"
  - "Linear Axis: <error>"
  - "Model: <error>"
  - "Storage: <error>"
  - "Light: <error>"
- [x] Error messages are descriptive
- [x] User can identify what to fix

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

## TEST SECTION 9: EDGE CASES & ADVANCED FEATURES

### TC-EDGE-001: Zero Detections
**Objective**: Verify behavior when no attachments detected
**Preconditions**: Empty image or no objects in view
**Steps**:
1. Capture image with no parts visible
2. Run detection
3. Observe behavior

**Expected Results**:
- [x] Step 1 completes with 0 detections
- [x] Detection table empty
- [x] Log shows "No detections found"
- [x] Steps 2-4 skipped (no work to do)
- [x] Result: OK (not an error, just empty)
- [x] Cycle time still recorded

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-EDGE-002: Many Detections (10+)
**Objective**: Verify system handles large detection counts
**Preconditions**: Image with 10+ attachments
**Steps**:
1. Use test image with many objects
2. Run inspection
3. Observe performance

**Expected Results**:
- [x] All detections processed
- [x] Table shows all entries
- [x] Motion occurs for all (may take 45-60 seconds)
- [x] No memory issues
- [x] No detection limit errors
- [x] All results saved

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-EDGE-003: Low Confidence Detections
**Objective**: Verify threshold filtering works correctly
**Preconditions**: Defect threshold set to 0.8 (high)
**Steps**:
1. Set defect score threshold to 0.8
2. Run inspection with marginal defects (0.5-0.7 confidence)
3. Review defect list

**Expected Results**:
- [x] Only defects >0.8 shown in ledger
- [x] Lower confidence detections filtered out
- [x] Threshold adjustable in UI (0.0-1.0)
- [x] Setting persists
- [x] Can re-run Step 4 with different threshold

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-EDGE-004: Edge/Contour Tuner
**Objective**: Verify contour tuner adjusts arrow computation
**Preconditions**: Main window open
**Steps**:
1. Click "Edge/Contour Tuner" button
2. Dialog opens with parameter controls
3. Load test image or capture top camera
4. Adjust parameters:
   - Blur kernel: 21 → 31
   - Threshold offset: 12 → 20
   - Approx epsilon: 1.0 → 2.0
5. Click "Preview Contour"
6. Click "Apply to Overlay"
7. Close dialog

**Expected Results**:
- [x] Dialog opens with all parameters
- [x] Can load image or capture
- [x] Preview updates in real-time (debounced 200ms)
- [x] Contour outline visible
- [x] Applying pushes to main preview
- [x] Parameters saved on close
- [x] Future detections use new params

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-EDGE-005: Parallel Motion Verification
**Objective**: Verify turntable and linear axis move simultaneously
**Preconditions**: Both axes connected and calibrated
**Steps**:
1. Start inspection with multiple detections
2. During Step 2, observe both axes
3. Verify simultaneous motion

**Expected Results**:
- [x] Turntable and axis start moving at same time
- [x] Both complete before front capture
- [x] No sequential wait (axis after turntable)
- [x] Time savings: ~2-3 seconds per detection
- [x] Motion coordinated safely

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-EDGE-006: Background Processing Verification
**Objective**: Verify Steps 3 & 4 run during Step 2 motion
**Preconditions**: Inspection with 3+ detections
**Steps**:
1. Start inspection
2. Monitor log output during Step 2
3. Look for Step 3 messages while turntable moving

**Expected Results**:
- [x] Log shows Step 3 processing while Step 2 moving to next
- [x] Step 4 messages appear during Step 2 motion
- [x] Parallel processing saves 10-15 seconds total
- [x] No race conditions or conflicts
- [x] Results consistent with sequential processing

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-EDGE-007: Image Upload (Manual)
**Objective**: Verify can run detection on uploaded image (no camera)
**Preconditions**: Step 2 - Upload Image section visible
**Steps**:
1. In Workflow tab, expand "Upload Image" section
2. Click "Browse" button
3. Select test image file
4. Run detection

**Expected Results**:
- [x] Can load image from disk
- [x] Image used instead of top camera capture
- [x] Step 1 runs normally
- [x] Steps 2-4 proceed if hardware available
- [x] Useful for offline testing or re-analysis

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-EDGE-008: FOV Calibration Accuracy
**Objective**: Verify linear axis positioning accuracy
**Preconditions**: Axis calibrated, known test pattern
**Steps**:
1. Place calibration target with known feature spacing
2. Run inspection
3. Measure linear axis positions vs expected
4. Verify front camera captures centered

**Expected Results**:
- [x] Linear axis moves to calculated positions
- [x] Front camera FOV centers on attachments
- [x] Accuracy: ±5mm or better
- [x] FOV calibration value: 951px (default)
- [x] Can adjust `front_fov_top_px` in config

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

## TEST SECTION 10: PERFORMANCE & USABILITY

### TC-PERF-001: Cycle Time - Standard Part
**Objective**: Benchmark typical inspection cycle time
**Preconditions**: Standard part with 4-6 attachments
**Steps**:
1. Run inspection on representative part
2. Record cycle time
3. Repeat 3 times, average

**Expected Results**:
- [x] Cycle time: 20-35 seconds (with GPU)
- [x] Breakdown approximate:
  - Step 1: 3-5 sec
  - Step 2: 12-20 sec (depends on detection count)
  - Step 3: 2-5 sec
  - Step 4: 2-5 sec
- [x] Consistent timing across runs (±10%)
- [x] Logged in cycle_time.txt

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-PERF-002: GPU Acceleration
**Objective**: Verify GPU provides performance benefit
**Preconditions**: System with CUDA GPU
**Steps**:
1. Run inspection with GPU enabled
2. Note cycle time
3. Disable GPU (uninstall CUDA or set device to CPU in code)
4. Run same inspection
5. Note cycle time

**Expected Results**:
- [x] GPU cycle time: 20-35 seconds
- [x] CPU cycle time: 60-120 seconds (3-4x slower)
- [x] Log shows "device: cuda" vs "device: cpu"
- [x] Both complete successfully
- [x] Results identical (performance only difference)

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-PERF-003: Memory Usage
**Objective**: Verify application memory footprint reasonable
**Preconditions**: Application running
**Steps**:
1. Open Task Manager or Resource Monitor
2. Note memory usage at idle
3. Run inspection
4. Note peak memory usage

**Expected Results**:
- [x] Idle: <2GB RAM
- [x] Peak (during inference): 3-6GB RAM (with GPU)
- [x] No memory leaks (stable after multiple runs)
- [x] Memory released after inspection completes
- [x] Can run multiple inspections without restart

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-PERF-004: UI Responsiveness
**Objective**: Verify UI remains responsive during inspection
**Preconditions**: Inspection running
**Steps**:
1. Start inspection
2. Attempt to interact with UI:
   - Scroll log
   - Resize window
   - Toggle overlays
   - View detection table
3. Observe responsiveness

**Expected Results**:
- [x] UI remains responsive (not frozen)
- [x] Can interact with controls
- [x] Log updates in real-time
- [x] Preview panes update as images captured
- [x] No "Not Responding" in window title

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-USABILITY-001: Initialization Wizard Flow
**Objective**: Verify wizard guides user logically through setup
**Preconditions**: Fresh installation
**Steps**:
1. Complete initialization wizard from start to finish
2. Evaluate flow and clarity

**Expected Results**:
- [x] Steps presented in logical order:
  1. Models (what to detect)
  2. Cameras (how to see)
  3. Preview (verify cameras work)
  4. Turntable (prepare motion)
- [x] Instructions clear at each step
- [x] Can navigate back/forward
- [x] Cannot proceed until current step valid
- [x] Validation messages helpful
- [x] "Begin Workflow" clear indication to start

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-USABILITY-002: Error Messages Clarity
**Objective**: Verify error messages are helpful to users
**Preconditions**: Various error conditions
**Steps**:
1. Trigger different errors
2. Evaluate message quality

**Expected Results**:
- [x] Messages state the problem clearly
- [x] Include what component failed
- [x] Suggest resolution when possible
- [x] Examples:
  - "Top camera not connected. Please connect camera and click Connect."
  - "Model file not found: C:/models/top.pth. Please check path and reload."
  - "Turntable not homed. Click Home button before starting inspection."
- [x] No cryptic error codes without explanation

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

### TC-USABILITY-003: Tooltips and Help
**Objective**: Verify tooltips provide guidance
**Preconditions**: Application running
**Steps**:
1. Hover over various controls
2. Check for tooltips

**Expected Results**:
- [x] Key controls have tooltips:
  - Connect buttons
  - Calibrate button
  - Run Detection button
  - Threshold spinners
- [x] Tooltips are concise and helpful
- [x] Appear after 1-2 second hover

**Status**: [ ] Pass [ ] Fail
**Notes**: ___________________________________________

---

## 5. Pass/Fail Criteria

### Overall UAT Acceptance Criteria

The system is **ACCEPTED** for production use if:

#### Critical Requirements (Must Pass 100%)
- [ ] All camera control tests pass (TC-CAM-001 to TC-CAM-008)
- [ ] All motion control tests pass (TC-TT-001 to TC-AXIS-007)
- [ ] All model loading tests pass (TC-MODEL-001 to TC-MODEL-008)
- [ ] All system health check tests pass (TC-HEALTH-001 to TC-HEALTH-006)
- [ ] Full inspection workflow executes successfully (TC-INSP-003)
- [ ] All results stored correctly with traceability (TC-RESULT-001 to TC-RESULT-005)
- [ ] Critical error handling works (TC-ERROR-001 to TC-ERROR-006)

#### Important Requirements (Must Pass ≥90%)
- [ ] Light controller tests (TC-LIGHT-001 to TC-LIGHT-005)
- [ ] Inspection control tests (TC-INSP-001 to TC-INSP-010)
- [ ] Edge case handling (TC-EDGE-001 to TC-EDGE-008)
- [ ] Performance meets targets (TC-PERF-001 to TC-PERF-004)

#### Nice-to-Have Requirements (Must Pass ≥80%)
- [ ] Usability tests (TC-USABILITY-001 to TC-USABILITY-003)
- [ ] Advanced features (TC-EDGE-004 to TC-EDGE-008)

### Test Summary

| Test Section | Total Tests | Passed | Failed | Pass Rate |
|--------------|-------------|--------|--------|-----------|
| Initialization | _ / _ | _ | _ | _% |
| Camera Control | _ / 8 | _ | _ | _% |
| Motion Control | _ / 13 | _ | _ | _% |
| Light Controller | _ / 5 | _ | _ | _% |
| Model Management | _ / 8 | _ | _ | _% |
| System Health | _ / 6 | _ | _ | _% |
| Main Inspection | _ / 10 | _ | _ | _% |
| Results & Traceability | _ / 5 | _ | _ | _% |
| Error Handling | _ / 6 | _ | _ | _% |
| Edge Cases | _ / 8 | _ | _ | _% |
| Performance & Usability | _ / 7 | _ | _ | _% |
| **TOTAL** | **_ / 76** | **_** | **_** | **_%** |

---

## 6. Sign-off

### Testing Team

**Tester Name**: ______________________________
**Role**: ______________________________
**Date**: ______________________________
**Signature**: ______________________________

**Notes/Comments**:
_______________________________________________
_______________________________________________
_______________________________________________

### Acceptance Decision

- [ ] **ACCEPTED** - System meets all critical requirements and is approved for production use
- [ ] **ACCEPTED WITH CONDITIONS** - System approved with noted deficiencies to be resolved in next release
- [ ] **REJECTED** - System does not meet acceptance criteria and requires remediation

**Decision By**: ______________________________
**Title**: ______________________________
**Date**: ______________________________
**Signature**: ______________________________

**Conditions/Action Items** (if applicable):
_______________________________________________
_______________________________________________
_______________________________________________

---

## Appendix A: Test Environment Details

**Test Date**: ______________________________
**Application Version**: ______________________________
**Test Location**: ______________________________

### Hardware Configuration
- **PC Model**: ______________________________
- **GPU**: ______________________________
- **Top Camera**: ______________________________
- **Front Camera**: ______________________________
- **Turntable**: ______________________________
- **Linear Axis**: ______________________________
- **LED Controller**: ______________________________

### Software Configuration
- **OS**: ______________________________
- **Python Version**: ______________________________
- **PyTorch Version**: ______________________________
- **Detectron2 Version**: ______________________________
- **CUDA Version**: ______________________________

### Model Configuration
- **Top Model**: ______________________________
- **Front Model**: ______________________________
- **Defect Model**: ______________________________

---

## Appendix B: Known Issues & Limitations

Document any known issues discovered during testing:

| Issue ID | Description | Severity | Workaround | Status |
|----------|-------------|----------|------------|--------|
| | | | | |
| | | | | |
| | | | | |

**Severity Levels**:
- **Critical**: Prevents core functionality, blocks production use
- **High**: Major impact, workaround available
- **Medium**: Moderate impact, acceptable for production
- **Low**: Minor annoyance, cosmetic issue

---

**End of UAT Document**
