# Factory Acceptance Test (FAT) Document
## Envista-Solomon Automated Inspection System

**Document Version**: 1.0
**FAT Date**: _______________
**Location**: Manufacturer Facility / Development Lab
**Purpose**: Pre-shipment validation and technical verification

---

## Table of Contents
1. [Introduction](#1-introduction)
2. [Test Objectives](#2-test-objectives)
3. [Test Environment](#3-test-environment)
4. [Pre-FAT Checklist](#4-pre-fat-checklist)
5. [FAT Test Procedures](#5-fat-test-procedures)
6. [Performance Benchmarks](#6-performance-benchmarks)
7. [Calibration & Configuration](#7-calibration--configuration)
8. [Documentation Verification](#8-documentation-verification)
9. [FAT Results Summary](#9-fat-results-summary)
10. [Sign-off & Shipping Authorization](#10-sign-off--shipping-authorization)

---

## 1. Introduction

### 1.1 Purpose
This Factory Acceptance Test (FAT) document defines the procedures for validating the Envista-Solomon automated inspection system prior to shipment to the customer site. The FAT ensures that all hardware, software, and functional requirements are met in a controlled factory environment.

### 1.2 Scope
The FAT covers:
- Hardware assembly and integration verification
- Software installation and configuration
- Functional testing of all subsystems
- Performance benchmarking against specifications
- Calibration and baseline measurements
- Documentation completeness

### 1.3 Test Location
**Facility**: _______________________________________________
**Address**: _______________________________________________
**Test Bay/Area**: _______________________________________________

### 1.4 Participants

#### Manufacturer Team
| Name | Role | Signature |
|------|------|-----------|
| | FAT Lead Engineer | |
| | Hardware Technician | |
| | Software Engineer | |
| | Quality Assurance | |

#### Customer Representatives (Optional)
| Name | Company | Role | Signature |
|------|---------|------|-----------|
| | | | |
| | | | |

---

## 2. Test Objectives

### 2.1 Primary Objectives
✅ Verify system meets all technical specifications
✅ Validate hardware integration and functionality
✅ Confirm software operates without critical defects
✅ Establish performance baselines
✅ Complete calibration and configuration
✅ Verify safety interlocks and emergency stops
✅ Ensure documentation is complete and accurate

### 2.2 Success Criteria
The system passes FAT if:
- All critical test cases pass (100%)
- Performance meets or exceeds specifications
- No critical or high-severity defects remain open
- All calibration within tolerance
- Documentation complete and accurate
- Safe for shipment and installation

### 2.3 Out of Scope
❌ Site-specific network integration (covered in SAT)
❌ Customer part testing (covered in SAT/UAT)
❌ Operator training (covered in SAT)
❌ Long-term reliability testing

---

## 3. Test Environment

### 3.1 System Configuration

#### Hardware Components
| Component | Model/Part Number | Serial Number | Status |
|-----------|-------------------|---------------|--------|
| **Main PC** | | | [ ] Verified |
| **GPU** | | | [ ] Verified |
| **Top Camera** | | | [ ] Verified |
| **Front Camera** | | | [ ] Verified |
| **Turntable** | | | [ ] Verified |
| **Linear Axis** | | | [ ] Verified |
| **LED Controller (ULC-2)** | | | [ ] Verified |
| **Power Supply Unit(s)** | | | [ ] Verified |
| **Emergency Stop** | | | [ ] Verified |
| **Enclosure/Frame** | | | [ ] Verified |

#### Software Configuration
| Software | Version | License | Status |
|----------|---------|---------|--------|
| **Windows OS** | | | [ ] Verified |
| **Python** | | | [ ] Verified |
| **PyTorch** | | | [ ] Verified |
| **Detectron2** | | | [ ] Verified |
| **Envista-Solomon App** | | | [ ] Verified |
| **Camera SDK (iRAYPLE)** | | | [ ] Verified |
| **Camera SDK (Pylon)** | | | [ ] Verified |
| **CUDA Toolkit** | | | [ ] Verified |

#### AI Models
| Model | Version | Classes | Trained On | Status |
|-------|---------|---------|------------|--------|
| **Top Attachment** | | | | [ ] Verified |
| **Front Attachment** | | | | [ ] Verified |
| **Defect Detection** | | | | [ ] Verified |

### 3.2 Test Parts/Fixtures
| Description | Quantity | Purpose | Status |
|-------------|----------|---------|--------|
| Golden Sample (Pass) | 1 | Baseline reference | [ ] Available |
| Known Defect Sample | 1 | Defect detection validation | [ ] Available |
| Calibration Target | 1 | FOV/positioning calibration | [ ] Available |
| Empty Fixture | 1 | Zero detection test | [ ] Available |
| Multi-feature Part (6+ items) | 1 | High-count stress test | [ ] Available |

### 3.3 Test Equipment
| Equipment | Model | Calibration Due | Purpose |
|-----------|-------|-----------------|---------|
| Multimeter | | | Power verification |
| Stopwatch/Timer | | | Cycle time measurement |
| Ruler/Caliper | | | Positioning accuracy |
| Network Cable Tester | | | Connectivity check |
| USB Cable (spare) | | | Camera connection backup |

---

## 4. Pre-FAT Checklist

### 4.1 Mechanical Assembly
- [ ] Frame/enclosure assembled and secured
- [ ] All mounting brackets installed
- [ ] Turntable mechanically installed
- [ ] Linear axis mechanically installed
- [ ] Camera mounts installed and leveled
- [ ] Light mounts installed
- [ ] Cable management completed
- [ ] Emergency stop button installed and accessible
- [ ] Covers/guards installed (if applicable)
- [ ] No sharp edges or pinch points

### 4.2 Electrical Installation
- [ ] Main power wiring complete
- [ ] All ground connections verified
- [ ] Fuses/circuit breakers rated correctly
- [ ] Power supply voltages measured and correct
- [ ] Emergency stop wired and functional
- [ ] Cable routing secure and protected
- [ ] No exposed conductors
- [ ] Electrical panel labeled

### 4.3 Communication/Network
- [ ] Top camera USB connected
- [ ] Front camera USB connected
- [ ] Turntable serial cable connected
- [ ] Linear axis serial cable connected
- [ ] LED controller Ethernet connected
- [ ] Network switch configured (if applicable)
- [ ] Static IP addresses assigned
- [ ] All cables labeled

### 4.4 Software Installation
- [ ] Operating system installed and updated
- [ ] All drivers installed (GPU, USB, serial)
- [ ] Python environment created
- [ ] All dependencies installed (requirements.txt)
- [ ] Camera SDKs installed
- [ ] Application code deployed
- [ ] AI models loaded and accessible
- [ ] user_settings.json configured
- [ ] Antivirus/firewall configured (if applicable)

### 4.5 Safety Verification
- [ ] Emergency stop tested (cuts power)
- [ ] No exposed moving parts
- [ ] Pinch points guarded
- [ ] Light intensity safe (no eye hazard)
- [ ] Electrical safety verified
- [ ] System can be safely powered off

---

## 5. FAT Test Procedures

---

## TEST SECTION 1: POWER & ELECTRICAL

### FAT-PWR-001: Power Supply Verification
**Objective**: Verify all power supplies deliver correct voltages
**Acceptance Criteria**: All voltages within ±5% of nominal

**Procedure**:
1. Ensure system powered OFF
2. Measure voltages at power supply outputs:
   - Main PC power: _____ V (Expected: 12V/5V/3.3V)
   - Turntable motor: _____ V (Expected: 24V)
   - Linear axis motor: _____ V (Expected: 24V)
   - LED controller: _____ V (Expected: 24V)
3. Power system ON
4. Re-measure voltages under load
5. Verify no voltage drops >5%

**Results**:
- [ ] Pass - All voltages within specification
- [ ] Fail - Out of tolerance: _______________________

**Notes**: _____________________________________________

---

### FAT-PWR-002: Emergency Stop Function
**Objective**: Verify E-stop immediately cuts power to motion systems
**Acceptance Criteria**: Motion stops within 500ms, no damage occurs

**Procedure**:
1. Power system ON
2. Initiate turntable motion (slow rotation)
3. Press emergency stop button
4. Verify turntable stops immediately
5. Verify PC remains powered (emergency stop does not cut PC power)
6. Reset E-stop
7. Repeat test with linear axis in motion

**Results**:
- [ ] Pass - Motion stops immediately, system safe
- [ ] Fail - Issue: _______________________

**Notes**: _____________________________________________

---

### FAT-PWR-003: Power-On Self Test (POST)
**Objective**: Verify system boots and initializes correctly
**Acceptance Criteria**: System boots to Windows desktop within 2 minutes

**Procedure**:
1. Ensure system powered OFF
2. Power ON
3. Observe boot sequence
4. Verify Windows desktop loads
5. Check no BIOS/hardware errors
6. Verify GPU detected (check Device Manager)

**Results**:
- Boot time: _____ seconds
- [ ] Pass - System boots normally
- [ ] Fail - Error: _______________________

**Notes**: _____________________________________________

---

## TEST SECTION 2: CAMERA SUBSYSTEM

### FAT-CAM-001: Camera Hardware Detection
**Objective**: Verify both cameras detected by SDK
**Acceptance Criteria**: Both cameras enumerate with unique identifiers

**Procedure**:
1. Launch Envista-Solomon application
2. Navigate to Camera Panel
3. Click "Refresh Devices" for Top Camera
4. Click "Refresh Devices" for Front Camera
5. Record camera details

**Results**:
- Top Camera:
  - Backend: [ ] iRAYPLE [ ] Pylon
  - Model: _______________________
  - Serial: _______________________
  - Index: _______
- Front Camera:
  - Backend: [ ] iRAYPLE [ ] Pylon
  - Model: _______________________
  - Serial: _______________________
  - Index: _______
- [ ] Pass - Both cameras detected
- [ ] Fail - Issue: _______________________

**Notes**: _____________________________________________

---

### FAT-CAM-002: Camera Connection Stability
**Objective**: Verify cameras connect reliably and maintain connection
**Acceptance Criteria**: 10/10 connect attempts succeed, no disconnects over 5 minutes

**Procedure**:
1. Disconnect both cameras
2. For each camera, perform 10 connection cycles:
   - Connect
   - Verify "Connected" status
   - Disconnect
   - Verify "Disconnected" status
3. Connect both cameras
4. Leave connected for 5 minutes
5. Verify both remain connected

**Results**:
- Top Camera: _____ / 10 successful connections
- Front Camera: _____ / 10 successful connections
- 5-minute stability: [ ] Both remained connected
- [ ] Pass - 10/10 connects, stable
- [ ] Fail - Issue: _______________________

**Notes**: _____________________________________________

---

### FAT-CAM-003: Image Capture Quality
**Objective**: Verify image quality meets specifications
**Acceptance Criteria**:
- Resolution: 2592x1944 or higher
- No dead pixels
- No excessive noise
- Proper exposure and focus

**Procedure**:
1. Connect both cameras
2. Place white calibration target in view
3. Capture image from top camera
4. Capture image from front camera
5. Inspect images for:
   - Resolution: Top _______ x _______, Front _______ x _______
   - Dead pixels: [ ] None detected
   - Noise level: [ ] Acceptable
   - Focus: [ ] Sharp
   - Exposure: [ ] Correct (not over/under exposed)
6. Save images to FAT report folder

**Results**:
- [ ] Pass - Image quality acceptable
- [ ] Fail - Issue: _______________________

**Captured Images**:
- Top: `fat_images/top_camera_quality.png`
- Front: `fat_images/front_camera_quality.png`

**Notes**: _____________________________________________

---

### FAT-CAM-004: Capture Frame Rate
**Objective**: Verify cameras can capture at required frame rate
**Acceptance Criteria**:
- Single capture completes in <2 seconds
- Buffer flush effective

**Procedure**:
1. Connect both cameras
2. Time 10 consecutive captures from top camera
3. Calculate average capture time
4. Time 10 consecutive captures from front camera
5. Test buffer flush (4 frames) completes in <500ms

**Results**:
- Top Camera Average: _____ ms per capture
- Front Camera Average: _____ ms per capture
- Buffer Flush Time: _____ ms
- [ ] Pass - Captures <2 sec, flush <500ms
- [ ] Fail - Issue: _______________________

**Notes**: _____________________________________________

---

### FAT-CAM-005: Camera Alignment
**Objective**: Verify cameras aligned to inspection area
**Acceptance Criteria**:
- Top camera centered on turntable (±10mm)
- Front camera horizontal, level

**Procedure**:
1. Place calibration target with crosshairs on turntable center
2. Capture top camera image
3. Measure crosshair position in image
4. Image center: (x: _____, y: _____)
5. Crosshair center: (x: _____, y: _____)
6. Offset: _____ mm (X), _____ mm (Y)
7. Place level target in front camera view
8. Verify horizon line is horizontal (±2°)

**Results**:
- Top Camera Alignment: [ ] Within ±10mm
- Front Camera Level: [ ] Within ±2°
- [ ] Pass - Cameras properly aligned
- [ ] Fail - Adjustment needed: _______________________

**Notes**: _____________________________________________

---

## TEST SECTION 3: MOTION CONTROL - TURNTABLE

### FAT-TT-001: Turntable Communication
**Objective**: Verify serial communication to turntable controller
**Acceptance Criteria**: Connection establishes, responds to commands

**Procedure**:
1. Connect turntable to PC via serial adapter
2. Identify COM port in Device Manager: COM_____
3. In Turntable Panel, select port and connect
4. Verify connection status: [ ] Connected
5. Send status query command
6. Verify response received within 1 second

**Results**:
- COM Port: _______
- Baud Rate: 115200
- Response Time: _____ ms
- [ ] Pass - Communication established
- [ ] Fail - Issue: _______________________

**Notes**: _____________________________________________

---

### FAT-TT-002: Turntable Homing Accuracy
**Objective**: Verify turntable homes to 0° position repeatably
**Acceptance Criteria**:
- Homing completes within 10 seconds
- Repeatability: ±0.5° over 10 cycles

**Procedure**:
1. Connect turntable
2. Perform 10 homing cycles
3. For each cycle:
   - Command Home
   - Wait for completion
   - Mark physical position on turntable and base
   - Record time to complete
4. Measure maximum deviation between marks

**Results**:
| Cycle | Time (s) | Position Match |
|-------|----------|----------------|
| 1 | | [ ] |
| 2 | | [ ] |
| 3 | | [ ] |
| 4 | | [ ] |
| 5 | | [ ] |
| 6 | | [ ] |
| 7 | | [ ] |
| 8 | | [ ] |
| 9 | | [ ] |
| 10 | | [ ] |

- Average Homing Time: _____ seconds
- Maximum Deviation: _____ degrees
- [ ] Pass - <10s, repeatability ±0.5°
- [ ] Fail - Issue: _______________________

**Notes**: _____________________________________________

---

### FAT-TT-003: Turntable Rotation Accuracy
**Objective**: Verify turntable rotates to commanded angles accurately
**Acceptance Criteria**: Position accuracy ±1° for all test angles

**Procedure**:
1. Home turntable
2. For each test angle, command rotation and measure actual:

| Commanded Angle | Actual Angle | Error | Status |
|----------------|--------------|-------|--------|
| 45° | _____ ° | _____ ° | [ ] Pass |
| 90° | _____ ° | _____ ° | [ ] Pass |
| 135° | _____ ° | _____ ° | [ ] Pass |
| 180° | _____ ° | _____ ° | [ ] Pass |
| 225° | _____ ° | _____ ° | [ ] Pass |
| 270° | _____ ° | _____ ° | [ ] Pass |
| 315° | _____ ° | _____ ° | [ ] Pass |
| 360° (0°) | _____ ° | _____ ° | [ ] Pass |

**Results**:
- Maximum Error: _____ degrees
- [ ] Pass - All within ±1°
- [ ] Fail - Issue: _______________________

**Notes**: _____________________________________________

---

### FAT-TT-004: Turntable Speed and Smoothness
**Objective**: Verify rotation is smooth without jerking or vibration
**Acceptance Criteria**:
- 360° rotation in 8-15 seconds
- No audible grinding or binding
- No visible vibration

**Procedure**:
1. Home turntable
2. Command 360° rotation
3. Time rotation: _____ seconds
4. Observe motion quality:
   - [ ] Smooth, constant speed
   - [ ] No jerking or sudden stops
   - [ ] No grinding noise
   - [ ] No excessive vibration
5. Repeat with part loaded (max weight): _____ kg

**Results**:
- Rotation Time (unloaded): _____ seconds
- Rotation Time (loaded): _____ seconds
- Motion Quality: [ ] Acceptable
- [ ] Pass - Smooth operation, 8-15s
- [ ] Fail - Issue: _______________________

**Notes**: _____________________________________________

---

### FAT-TT-005: Turntable Bidirectional Operation
**Objective**: Verify turntable operates correctly in both directions
**Acceptance Criteria**: CW and CCW rotation equally smooth and accurate

**Procedure**:
1. Home turntable
2. Rotate CW 180°: Time = _____ s, Smooth: [ ]
3. Rotate CCW 180° (back to 0°): Time = _____ s, Smooth: [ ]
4. Rotate CCW 90° (-90°): Actual = _____ °
5. Rotate CW 90° (back to 0°): Actual = _____ °
6. Verify no backlash or dead zone

**Results**:
- CW Performance: [ ] Acceptable
- CCW Performance: [ ] Acceptable
- Backlash: _____ degrees (should be <0.5°)
- [ ] Pass - Bidirectional operation correct
- [ ] Fail - Issue: _______________________

**Notes**: _____________________________________________

---

## TEST SECTION 4: MOTION CONTROL - LINEAR AXIS

### FAT-AXIS-001: Linear Axis Communication
**Objective**: Verify serial communication to axis controller
**Acceptance Criteria**: Connection establishes, responds to commands

**Procedure**:
1. Connect linear axis to PC via serial adapter
2. Identify COM port: COM_____
3. In Linear Axis Panel, select port and connect
4. Verify connection status: [ ] Connected
5. Send status query command
6. Verify response received within 1 second

**Results**:
- COM Port: _______
- Baud Rate: 9600
- Response Time: _____ ms
- [ ] Pass - Communication established
- [ ] Fail - Issue: _______________________

**Notes**: _____________________________________________

---

### FAT-AXIS-002: Linear Axis Calibration
**Objective**: Verify auto-calibration correctly detects travel limits
**Acceptance Criteria**:
- Calibration completes successfully
- Travel range: 100mm ±2mm

**Procedure**:
1. Connect axis
2. Clear any previous calibration
3. Click "Calibrate" button
4. Observe calibration sequence:
   - [ ] Moves to left limit
   - [ ] Moves to right limit
   - [ ] Returns to center
   - [ ] Button turns green
5. Measure physical travel with ruler: _____ mm

**Results**:
- Calibration Time: _____ seconds
- Detected Travel Range: _____ mm
- Physical Travel Range: _____ mm
- [ ] Pass - Calibration successful, 100mm ±2mm
- [ ] Fail - Issue: _______________________

**Notes**: _____________________________________________

---

### FAT-AXIS-003: Linear Axis Positioning Accuracy
**Objective**: Verify axis moves to commanded positions accurately
**Acceptance Criteria**: Position accuracy ±0.5mm for all test positions

**Procedure**:
1. Calibrate axis
2. For each test position, command move and measure actual:

| Commanded Position | Actual Position | Error | Status |
|-------------------|-----------------|-------|--------|
| 0 mm | _____ mm | _____ mm | [ ] Pass |
| 10 mm | _____ mm | _____ mm | [ ] Pass |
| 25 mm | _____ mm | _____ mm | [ ] Pass |
| 50 mm | _____ mm | _____ mm | [ ] Pass |
| 75 mm | _____ mm | _____ mm | [ ] Pass |
| 90 mm | _____ mm | _____ mm | [ ] Pass |
| 100 mm | _____ mm | _____ mm | [ ] Pass |

**Results**:
- Maximum Error: _____ mm
- [ ] Pass - All within ±0.5mm
- [ ] Fail - Issue: _______________________

**Notes**: _____________________________________________

---

### FAT-AXIS-004: Linear Axis Repeatability
**Objective**: Verify axis returns to same position repeatedly
**Acceptance Criteria**: Repeatability ±0.2mm over 10 cycles to 50mm position

**Procedure**:
1. Calibrate axis
2. Perform 10 cycles to 50mm position:

| Cycle | Position Reading | Deviation from 50mm |
|-------|------------------|---------------------|
| 1 | _____ mm | _____ mm |
| 2 | _____ mm | _____ mm |
| 3 | _____ mm | _____ mm |
| 4 | _____ mm | _____ mm |
| 5 | _____ mm | _____ mm |
| 6 | _____ mm | _____ mm |
| 7 | _____ mm | _____ mm |
| 8 | _____ mm | _____ mm |
| 9 | _____ mm | _____ mm |
| 10 | _____ mm | _____ mm |

**Results**:
- Maximum Deviation: _____ mm
- [ ] Pass - Repeatability ±0.2mm
- [ ] Fail - Issue: _______________________

**Notes**: _____________________________________________

---

### FAT-AXIS-005: Linear Axis Speed
**Objective**: Verify axis moves at acceptable speed
**Acceptance Criteria**: Full 100mm travel in 3-8 seconds

**Procedure**:
1. Calibrate axis
2. Command move from 0mm to 100mm
3. Time motion: _____ seconds
4. Command move from 100mm to 0mm
5. Time motion: _____ seconds
6. Calculate average speed: _____ mm/s

**Results**:
- Forward Travel Time: _____ seconds
- Reverse Travel Time: _____ seconds
- Average Speed: _____ mm/s
- [ ] Pass - Travel time 3-8 seconds
- [ ] Fail - Issue: _______________________

**Notes**: _____________________________________________

---

## TEST SECTION 5: LIGHT CONTROLLER

### FAT-LIGHT-001: LED Controller Network Connection
**Objective**: Verify LED controller reachable on network
**Acceptance Criteria**: Ping successful, UDP communication established

**Procedure**:
1. Connect LED controller to network
2. Configure IP address: ___.___.___.___
3. From PC, ping LED controller:
   - Command: `ping [IP] -n 10`
   - Packets sent: 10
   - Packets received: _____
   - Packet loss: _____%
4. In application, enter IP and connect
5. Verify status shows "Online"

**Results**:
- IP Address: ___.___.___.___
- Ping Success Rate: _____%
- Connection Status: [ ] Online
- [ ] Pass - Network communication OK
- [ ] Fail - Issue: _______________________

**Notes**: _____________________________________________

---

### FAT-LIGHT-002: Light Intensity Control
**Objective**: Verify light intensity adjusts correctly
**Acceptance Criteria**:
- Intensity adjustable from 0-250mA
- Brightness visibly changes
- Settings persist

**Procedure**:
1. Connect LED controller
2. Set top light to 0 mA: [ ] Light OFF
3. Set top light to 50 mA: [ ] Dim visible
4. Set top light to 150 mA: [ ] Medium bright
5. Set top light to 250 mA: [ ] Maximum bright
6. Measure actual current with multimeter:
   - At 50 mA setting: _____ mA measured
   - At 150 mA setting: _____ mA measured
   - At 250 mA setting: _____ mA measured
7. Verify read-back matches setting

**Results**:
- Current Accuracy: [ ] Within ±10%
- Brightness Changes Visible: [ ] Yes
- Settings Persist: [ ] Yes
- [ ] Pass - Light control functional
- [ ] Fail - Issue: _______________________

**Notes**: _____________________________________________

---

### FAT-LIGHT-003: Independent Channel Control
**Objective**: Verify top and front lights control independently
**Acceptance Criteria**: Channels operate independently without cross-talk

**Procedure**:
1. Set top light to 200 mA
2. Set front light to 100 mA
3. Verify both lights at correct intensity
4. Change top to 50 mA
5. Verify front remains at 100 mA
6. Change front to 200 mA
7. Verify top remains at 50 mA

**Results**:
- Independent Control: [ ] Confirmed
- No Cross-talk: [ ] Confirmed
- [ ] Pass - Independent operation
- [ ] Fail - Issue: _______________________

**Notes**: _____________________________________________

---

## TEST SECTION 6: AI MODEL VALIDATION

### FAT-MODEL-001: Model Loading
**Objective**: Verify all three models load successfully
**Acceptance Criteria**:
- All models load without errors
- Class names and colors parsed correctly
- GPU acceleration detected

**Procedure**:
1. Launch application
2. Load Top Attachment Model:
   - Path: _______________________________________
   - Status: [ ] Loaded
   - Classes: _______________________________________
   - Device: [ ] cuda [ ] cpu
3. Load Front Attachment Model:
   - Path: _______________________________________
   - Status: [ ] Loaded
   - Classes: _______________________________________
   - Device: [ ] cuda [ ] cpu
4. Load Defect Detection Model:
   - Path: _______________________________________
   - Status: [ ] Loaded
   - Classes: _______________________________________
   - Device: [ ] cuda [ ] cpu

**Results**:
- [ ] Pass - All models loaded successfully
- [ ] Fail - Issue: _______________________

**Notes**: _____________________________________________

---

### FAT-MODEL-002: Inference Speed - Top Model
**Objective**: Benchmark top model inference speed
**Acceptance Criteria**:
- GPU inference: <2 seconds per image
- CPU inference: <10 seconds per image

**Procedure**:
1. Load top model
2. Use test image (2592x1944)
3. Run inference 5 times
4. Record inference times:
   - Run 1: _____ seconds
   - Run 2: _____ seconds
   - Run 3: _____ seconds
   - Run 4: _____ seconds
   - Run 5: _____ seconds
5. Calculate average: _____ seconds

**Results**:
- Average Inference Time: _____ seconds
- Device Used: [ ] GPU [ ] CPU
- [ ] Pass - Meets timing requirement
- [ ] Fail - Too slow: _______________________

**Notes**: _____________________________________________

---

### FAT-MODEL-003: Inference Speed - Front Model
**Objective**: Benchmark front model inference speed
**Acceptance Criteria**:
- GPU inference: <2 seconds per image
- CPU inference: <10 seconds per image

**Procedure**:
1. Load front model
2. Use test crop image
3. Run inference 5 times
4. Record inference times:
   - Run 1: _____ seconds
   - Run 2: _____ seconds
   - Run 3: _____ seconds
   - Run 4: _____ seconds
   - Run 5: _____ seconds
5. Calculate average: _____ seconds

**Results**:
- Average Inference Time: _____ seconds
- Device Used: [ ] GPU [ ] CPU
- [ ] Pass - Meets timing requirement
- [ ] Fail - Too slow: _______________________

**Notes**: _____________________________________________

---

### FAT-MODEL-004: Inference Speed - Defect Model
**Objective**: Benchmark defect model inference speed
**Acceptance Criteria**:
- GPU inference: <2 seconds per image
- CPU inference: <10 seconds per image

**Procedure**:
1. Load defect model
2. Use test bbox crop
3. Run inference 5 times
4. Record inference times:
   - Run 1: _____ seconds
   - Run 2: _____ seconds
   - Run 3: _____ seconds
   - Run 4: _____ seconds
   - Run 5: _____ seconds
5. Calculate average: _____ seconds

**Results**:
- Average Inference Time: _____ seconds
- Device Used: [ ] GPU [ ] CPU
- [ ] Pass - Meets timing requirement
- [ ] Fail - Too slow: _______________________

**Notes**: _____________________________________________

---

### FAT-MODEL-005: Detection Accuracy - Golden Sample
**Objective**: Verify models detect known-good sample correctly
**Acceptance Criteria**:
- All expected attachments detected
- No false positives
- Confidence scores >0.7

**Procedure**:
1. Use golden sample part
2. Run full inspection workflow
3. Record results:
   - Expected detections: _____
   - Actual detections: _____
   - False positives: _____
   - Confidence scores: _______________________
4. Review annotated images
5. Verify defect model reports PASS

**Results**:
- Detection Rate: _____ / _____ (100% expected)
- False Positives: _____
- Overall Result: [ ] PASS [ ] FAIL
- [ ] Pass - Golden sample detected correctly
- [ ] Fail - Issue: _______________________

**Notes**: _____________________________________________

---

### FAT-MODEL-006: Detection Accuracy - Defect Sample
**Objective**: Verify defect model detects known defects
**Acceptance Criteria**:
- Known defects detected
- Confidence scores >0.5

**Procedure**:
1. Use known defect sample part
2. Run full inspection workflow
3. Record defect results:
   - Expected defects: _____ (type: _______)
   - Detected defects: _____ (type: _______)
   - Confidence scores: _______________________
4. Verify defect model reports FAIL
5. Review defect annotations

**Results**:
- Defect Detection Rate: _____ / _____ (100% expected)
- False Negatives: _____
- Overall Result: [ ] FAIL (as expected)
- [ ] Pass - Defects detected correctly
- [ ] Fail - Issue: _______________________

**Notes**: _____________________________________________

---

## TEST SECTION 7: INTEGRATED SYSTEM TESTS

### FAT-SYS-001: Full Inspection Workflow - Golden Sample
**Objective**: Verify complete 4-step workflow executes successfully
**Acceptance Criteria**:
- Workflow completes without errors
- Cycle time <45 seconds
- All data saved correctly

**Procedure**:
1. Place golden sample on turntable
2. Enter Part ID: "FAT_Golden_001"
3. Click "Run Detection"
4. Monitor workflow execution
5. Record results:
   - Step 1 detections: _____
   - Step 2 rotations: _____
   - Step 3 front detections: _____
   - Step 4 defects: _____ (should be 0)
   - Cycle time: _____ seconds
6. Verify images saved to `captures/`

**Results**:
- Workflow Completion: [ ] Success
- Cycle Time: _____ seconds
- Data Saved: [ ] Yes
- Overall Result: [ ] PASS
- [ ] Pass - Golden sample workflow OK
- [ ] Fail - Issue: _______________________

**Saved Data Path**: _______________________________________

**Notes**: _____________________________________________

---

### FAT-SYS-002: Full Inspection Workflow - Defect Sample
**Objective**: Verify workflow detects defects correctly
**Acceptance Criteria**:
- Workflow completes
- Defects detected and logged
- Overall result: FAIL

**Procedure**:
1. Place defect sample on turntable
2. Enter Part ID: "FAT_Defect_001"
3. Click "Run Detection"
4. Monitor workflow execution
5. Record results:
   - Step 1 detections: _____
   - Step 2 rotations: _____
   - Step 3 front detections: _____
   - Step 4 defects: _____ (should be >0)
   - Defect types: _______________________
   - Cycle time: _____ seconds

**Results**:
- Workflow Completion: [ ] Success
- Defects Detected: _____
- Overall Result: [ ] FAIL (as expected)
- [ ] Pass - Defect detection workflow OK
- [ ] Fail - Issue: _______________________

**Notes**: _____________________________________________

---

### FAT-SYS-003: Multi-Feature Part Stress Test
**Objective**: Verify system handles high detection count
**Acceptance Criteria**:
- All 6+ detections processed
- Cycle time <60 seconds
- No memory issues

**Procedure**:
1. Place multi-feature part (6+ attachments)
2. Enter Part ID: "FAT_Stress_001"
3. Click "Run Detection"
4. Monitor system resources during execution
5. Record results:
   - Step 1 detections: _____
   - Cycle time: _____ seconds
   - Peak RAM usage: _____ GB
   - Peak GPU usage: _____%

**Results**:
- Detection Count: _____
- Cycle Time: _____ seconds
- System Stability: [ ] Stable
- [ ] Pass - Handles high count correctly
- [ ] Fail - Issue: _______________________

**Notes**: _____________________________________________

---

### FAT-SYS-004: Empty Fixture Test (Zero Detections)
**Objective**: Verify system handles zero detections gracefully
**Acceptance Criteria**:
- No errors or crashes
- Reports zero detections
- Steps 2-4 skipped

**Procedure**:
1. Place empty fixture (no parts)
2. Enter Part ID: "FAT_Empty_001"
3. Click "Run Detection"
4. Monitor workflow
5. Verify behavior:
   - Step 1: _____ detections (should be 0)
   - Step 2: [ ] Skipped
   - Step 3: [ ] Skipped
   - Step 4: [ ] Skipped
   - Overall result: [ ] PASS

**Results**:
- Zero Detection Handling: [ ] Correct
- No Errors: [ ] Confirmed
- [ ] Pass - Empty fixture handled correctly
- [ ] Fail - Issue: _______________________

**Notes**: _____________________________________________

---

### FAT-SYS-005: Parallel Processing Verification
**Objective**: Verify Steps 3 & 4 execute during Step 2 motion
**Acceptance Criteria**: Background processing saves ≥10 seconds

**Procedure**:
1. Use part with 4+ detections
2. Run inspection
3. Monitor log output during Step 2
4. Verify Step 3 messages appear while turntable moving
5. Compare to theoretical sequential time:
   - Motion time per detection: _____ s × _____ detections = _____ s
   - Step 3 time per detection: _____ s × _____ detections = _____ s
   - Step 4 time per detection: _____ s × _____ detections = _____ s
   - Sequential total: _____ s
   - Actual total: _____ s
   - Time saved: _____ s

**Results**:
- Parallel Processing Confirmed: [ ] Yes
- Time Saved: _____ seconds
- [ ] Pass - Parallel processing working
- [ ] Fail - Sequential execution detected

**Notes**: _____________________________________________

---

### FAT-SYS-006: Coordinated Motion Test
**Objective**: Verify turntable and linear axis move simultaneously
**Acceptance Criteria**: Both axes start and complete within 1 second of each other

**Procedure**:
1. Run inspection with part requiring axis positioning
2. Observe motion during Step 2
3. For one detection, time:
   - Turntable start: t = 0
   - Linear axis start: t = _____ s (should be ~0s)
   - Turntable complete: t = _____ s
   - Linear axis complete: t = _____ s
   - Difference: _____ s (should be <1s)

**Results**:
- Simultaneous Start: [ ] Yes (within 1s)
- Near-Simultaneous Completion: [ ] Yes
- [ ] Pass - Coordinated motion confirmed
- [ ] Fail - Sequential motion detected

**Notes**: _____________________________________________

---

### FAT-SYS-007: Crash Recovery Test
**Objective**: Verify crash logging captures errors
**Acceptance Criteria**: Crash logged, system can restart cleanly

**Procedure**:
1. Note: This test may be skipped if no safe way to trigger crash
2. If possible, trigger controlled exception
3. Check for `crash.log` file
4. Verify log contains:
   - [ ] Timestamp
   - [ ] Exception type
   - [ ] Stack trace
5. Restart application
6. Verify system recovers cleanly

**Results**:
- Crash Log Created: [ ] Yes [ ] N/A
- Log Complete: [ ] Yes [ ] N/A
- Clean Restart: [ ] Yes
- [ ] Pass - Crash logging functional
- [ ] N/A - Test skipped (system stable)

**Notes**: _____________________________________________

---

## 6. Performance Benchmarks

### 6.1 Cycle Time Benchmarks

| Test Scenario | Detection Count | Target Time | Actual Time | Status |
|--------------|-----------------|-------------|-------------|--------|
| Golden Sample | 4 | <30s | _____ s | [ ] Pass |
| Defect Sample | 4 | <30s | _____ s | [ ] Pass |
| Stress Test | 6+ | <60s | _____ s | [ ] Pass |
| Empty Fixture | 0 | <10s | _____ s | [ ] Pass |

**Overall Cycle Time Performance**: [ ] Meets Specifications

---

### 6.2 Hardware Performance

| Parameter | Specification | Measured | Status |
|-----------|---------------|----------|--------|
| **Turntable** | | | |
| Homing Time | <10s | _____ s | [ ] Pass |
| 360° Rotation | 8-15s | _____ s | [ ] Pass |
| Position Accuracy | ±1° | ±_____ ° | [ ] Pass |
| Repeatability | ±0.5° | ±_____ ° | [ ] Pass |
| **Linear Axis** | | | |
| Calibration Time | <30s | _____ s | [ ] Pass |
| Full Travel (100mm) | 3-8s | _____ s | [ ] Pass |
| Position Accuracy | ±0.5mm | ±_____ mm | [ ] Pass |
| Repeatability | ±0.2mm | ±_____ mm | [ ] Pass |
| **Cameras** | | | |
| Resolution (Top) | ≥2592x1944 | _____ x _____ | [ ] Pass |
| Resolution (Front) | ≥2592x1944 | _____ x _____ | [ ] Pass |
| Capture Time | <2s | _____ s | [ ] Pass |
| **AI Models** | | | |
| Top Model Inference | <2s (GPU) | _____ s | [ ] Pass |
| Front Model Inference | <2s (GPU) | _____ s | [ ] Pass |
| Defect Model Inference | <2s (GPU) | _____ s | [ ] Pass |

**Overall Hardware Performance**: [ ] Meets Specifications

---

### 6.3 System Resource Usage

| Resource | Idle | Peak (Inspection) | Limit | Status |
|----------|------|-------------------|-------|--------|
| CPU Usage | ____% | ____% | <80% | [ ] Pass |
| RAM Usage | ____ GB | ____ GB | <8GB | [ ] Pass |
| GPU Usage | ____% | ____% | <95% | [ ] Pass |
| GPU Memory | ____ GB | ____ GB | <6GB | [ ] Pass |
| Disk I/O | Low | ____ MB/s | <100 MB/s | [ ] Pass |

**Overall Resource Usage**: [ ] Within Limits

---

## 7. Calibration & Configuration

### 7.1 Calibration Records

#### Camera Calibration
- **Top Camera**:
  - Alignment to turntable center: (_____, _____)
  - Offset: ±_____ mm
  - Focus setting: _______
  - Exposure: _______ ms
  - Calibration Date: _______
  - Calibrated By: _______

- **Front Camera**:
  - Level: ±_____ degrees
  - Focus setting: _______
  - Exposure: _______ ms
  - FOV width in top image: _____ pixels (default 951)
  - Calibration Date: _______
  - Calibrated By: _______

#### Motion Calibration
- **Turntable**:
  - Home position marked: [ ] Yes
  - Zero offset: _____ degrees
  - Direction: [ ] CW [ ] CCW
  - Calibration Date: _______
  - Calibrated By: _______

- **Linear Axis**:
  - Travel range: _____ mm
  - Home position: _____ mm
  - Position accuracy verified: [ ] Yes
  - Calibration Date: _______
  - Calibrated By: _______

#### Light Calibration
- **LED Controller**:
  - Top light optimal setting: _____ mA
  - Front light optimal setting: _____ mA
  - Dwell time: _____ ms
  - Calibration Date: _______
  - Calibrated By: _______

### 7.2 Configuration File Backup

- [ ] `user_settings.json` backed up to FAT documentation folder
- [ ] Model metadata files (`model_final.json`) backed up
- [ ] Calibration images saved
- Backup Location: _______________________________________

---

## 8. Documentation Verification

### 8.1 Technical Documentation

- [ ] System schematic provided
- [ ] Wiring diagram provided
- [ ] Bill of Materials (BOM) provided
- [ ] Component datasheets included
- [ ] Software installation guide provided
- [ ] Model training documentation provided (if applicable)

### 8.2 User Documentation

- [ ] User Manual provided
- [ ] Quick Start Guide provided
- [ ] Troubleshooting Guide provided
- [ ] Safety Instructions provided
- [ ] Maintenance Schedule provided

### 8.3 Test Documentation

- [ ] This FAT Document completed
- [ ] All test results recorded
- [ ] Test images/videos archived
- [ ] Performance benchmark data recorded
- [ ] Calibration records completed

### 8.4 Compliance Documentation

- [ ] CE Marking documentation (if applicable)
- [ ] Electrical safety certification
- [ ] RoHS/REACH compliance
- [ ] Export control classification
- [ ] Software licenses documented

---

## 9. FAT Results Summary

### 9.1 Test Summary by Section

| Test Section | Total Tests | Passed | Failed | N/A | Pass Rate |
|--------------|-------------|--------|--------|-----|-----------|
| Power & Electrical | 3 | ___ | ___ | ___ | ___% |
| Camera Subsystem | 5 | ___ | ___ | ___ | ___% |
| Motion - Turntable | 5 | ___ | ___ | ___ | ___% |
| Motion - Linear Axis | 5 | ___ | ___ | ___ | ___% |
| Light Controller | 3 | ___ | ___ | ___ | ___% |
| AI Model Validation | 6 | ___ | ___ | ___ | ___% |
| Integrated System | 7 | ___ | ___ | ___ | ___% |
| **TOTAL** | **34** | **___** | **___** | **___** | **___%** |

### 9.2 Critical Issues (Must Be Resolved Before Shipment)

| Issue ID | Description | Severity | Status | Resolution |
|----------|-------------|----------|--------|------------|
| | | | | |
| | | | | |

### 9.3 Minor Issues (Can Be Resolved On-Site or Later)

| Issue ID | Description | Severity | Status | Notes |
|----------|-------------|----------|--------|-------|
| | | | | |
| | | | | |

### 9.4 Recommendations for Site Installation

1. _____________________________________________
2. _____________________________________________
3. _____________________________________________
4. _____________________________________________

### 9.5 Special Shipping Instructions

- [ ] Protect camera lenses during transport
- [ ] Secure all moving parts (turntable, axis)
- [ ] Include calibration targets and test parts
- [ ] Ship models on USB drive (backup)
- [ ] Include FAT documentation package
- [ ] Fragile labels applied to packaging

**Shipping Notes**: _____________________________________________
_____________________________________________

---

## 10. Sign-off & Shipping Authorization

### 10.1 FAT Completion

**FAT Start Date**: _______________________
**FAT End Date**: _______________________
**Total Duration**: _______ days

### 10.2 Test Team Sign-off

I certify that all FAT test procedures have been executed according to this document and that the results have been accurately recorded.

**FAT Lead Engineer**:
- Name: _______________________________
- Signature: ___________________________
- Date: _______________________________

**Hardware Technician**:
- Name: _______________________________
- Signature: ___________________________
- Date: _______________________________

**Software Engineer**:
- Name: _______________________________
- Signature: ___________________________
- Date: _______________________________

**Quality Assurance**:
- Name: _______________________________
- Signature: ___________________________
- Date: _______________________________

### 10.3 Customer Witness Sign-off (If Present)

I have witnessed the Factory Acceptance Test and confirm that the system meets the specified requirements.

**Customer Representative**:
- Name: _______________________________
- Company: _____________________________
- Signature: ___________________________
- Date: _______________________________

**Comments**: _____________________________________________
_____________________________________________

### 10.4 FAT Result

- [ ] **PASS** - System approved for shipment
- [ ] **CONDITIONAL PASS** - System approved with conditions (see Section 9.2)
- [ ] **FAIL** - System requires rework before shipment

**Conditions (if applicable)**: _____________________________________________
_____________________________________________

### 10.5 Shipping Authorization

Based on the FAT results, I authorize this system for shipment to the customer site.

**Authorized By**:
- Name: _______________________________
- Title: _______________________________
- Company: _____________________________
- Signature: ___________________________
- Date: _______________________________

**Expected Ship Date**: _______________________
**Destination**: _____________________________________________

---

## Appendix A: Test Equipment Calibration Records

| Equipment | Serial Number | Last Calibration | Next Calibration | Cert # |
|-----------|---------------|------------------|------------------|--------|
| | | | | |
| | | | | |

---

## Appendix B: Test Images and Videos

All test images and videos archived to:
**Location**: _____________________________________________

**Contents**:
- [ ] Camera quality test images
- [ ] Calibration images (top and front)
- [ ] Golden sample inspection images (full set)
- [ ] Defect sample inspection images (full set)
- [ ] Motion test videos (turntable, axis)
- [ ] Workflow execution video

---

## Appendix C: Configuration Backups

All configuration files backed up to:
**Location**: _____________________________________________

**Contents**:
- [ ] `user_settings.json`
- [ ] Model metadata files (3x `model_final.json`)
- [ ] Camera SDK configuration (if applicable)
- [ ] Network configuration (static IPs)

---

## Appendix D: Defect Log

All defects discovered during FAT:

| ID | Date | Severity | Component | Description | Resolution | Closed |
|----|------|----------|-----------|-------------|------------|--------|
| | | | | | | [ ] |
| | | | | | | [ ] |
| | | | | | | [ ] |

**Severity Levels**:
- **Critical**: System inoperable, blocks shipment
- **High**: Major functionality impaired, must fix before or immediately after shipment
- **Medium**: Moderate issue, workaround available, fix during SAT
- **Low**: Minor issue, does not affect operation

---

**END OF FAT DOCUMENT**

**Document Control**:
- Document ID: FAT-EnvistaSolomon-001
- Version: 1.0
- Created By: _______________________________
- Approved By: _______________________________
- Date: _______________________________
