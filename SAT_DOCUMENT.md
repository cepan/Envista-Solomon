# Site Acceptance Test (SAT) Document
## Envista-Solomon Automated Inspection System

**Document Version**: 1.0
**SAT Date**: _______________
**Location**: Customer Production Facility
**Purpose**: On-site validation after installation

---

## Table of Contents
1. [Introduction](#1-introduction)
2. [Test Objectives](#2-test-objectives)
3. [Site Information](#3-site-information)
4. [Pre-SAT Checklist](#4-pre-sat-checklist)
5. [SAT Test Procedures](#5-sat-test-procedures)
6. [Site Integration Testing](#6-site-integration-testing)
7. [Operator Training Verification](#7-operator-training-verification)
8. [Production Readiness](#8-production-readiness)
9. [SAT Results Summary](#9-sat-results-summary)
10. [Sign-off & Handover](#10-sign-off--handover)

---

## 1. Introduction

### 1.1 Purpose
This Site Acceptance Test (SAT) document defines the procedures for validating the Envista-Solomon automated inspection system after installation at the customer production facility. The SAT confirms that the system operates correctly in the actual production environment and is ready for operational use.

### 1.2 Scope
The SAT covers:
- Post-shipment hardware verification
- Site-specific integration (network, power, layout)
- Testing with customer's actual production parts
- Environmental validation (lighting, temperature, vibration)
- Operator training and competency verification
- Production workflow integration
- Final acceptance for production use

### 1.3 Relationship to FAT
This SAT assumes:
- Factory Acceptance Test (FAT) completed successfully
- System shipped and received without damage
- FAT documentation available for reference

### 1.4 Test Location
**Customer Facility**: _______________________________________________
**Address**: _______________________________________________
**Production Line/Area**: _______________________________________________
**Installation Date**: _______________________________________________

### 1.5 Participants

#### Installation Team (Manufacturer)
| Name | Role | Contact | Signature |
|------|------|---------|-----------|
| | Field Service Engineer | | |
| | Software Support | | |
| | Training Specialist | | |

#### Customer Team
| Name | Role | Department | Signature |
|------|------|------------|-----------|
| | Production Manager | | |
| | Quality Engineer | | |
| | Operator (Primary) | | |
| | Operator (Secondary) | | |
| | IT/Network Admin | | |
| | Maintenance Technician | | |

---

## 2. Test Objectives

### 2.1 Primary Objectives
✅ Verify system survived shipping without damage
✅ Validate integration with site infrastructure
✅ Confirm operation with customer's actual parts
✅ Test environmental conditions are adequate
✅ Train operators and verify competency
✅ Integrate with customer's production workflow
✅ Establish site-specific baselines and settings
✅ Obtain final customer acceptance for production use

### 2.2 Success Criteria
The system passes SAT if:
- All hardware functions correctly after installation
- Site infrastructure (power, network, space) is adequate
- System inspects customer parts meeting quality standards
- Cycle time meets production requirements
- Operators trained and competent
- Integration with workflow successful
- Customer approves for production use

### 2.3 Out of Scope
❌ Hardware repairs (should have been resolved in FAT)
❌ Major software modifications (requires development cycle)
❌ Model re-training (separate activity)
❌ Long-term production monitoring (covered in production phase)

---

## 3. Site Information

### 3.1 Facility Details
**Production Line**: _______________________
**Shift Schedule**: _______________________
**Daily Production Volume**: _______ parts/day
**Target Inspection Rate**: _______ parts/hour
**Part Types to Inspect**: _______________________

### 3.2 Environmental Conditions
| Parameter | Specification | Measured | Status |
|-----------|---------------|----------|--------|
| **Temperature** | 15-30°C | _____ °C | [ ] OK |
| **Humidity** | 30-80% RH | _____ % | [ ] OK |
| **Ambient Light** | <500 lux | _____ lux | [ ] OK |
| **Vibration** | <0.5g | _____ g | [ ] OK |
| **Dust Level** | Clean area | Visual: ____ | [ ] OK |

### 3.3 Power Infrastructure
| Parameter | Requirement | Available | Status |
|-----------|-------------|-----------|--------|
| **Voltage** | 110-120V or 220-240V | _____ V | [ ] OK |
| **Frequency** | 50/60 Hz | _____ Hz | [ ] OK |
| **Power Capacity** | 1500W | _____ W | [ ] OK |
| **Ground** | Proper earth ground | Measured: ____ Ω | [ ] OK |
| **Circuit Breaker** | 20A dedicated | _____ A | [ ] OK |
| **Surge Protection** | Recommended | [ ] Installed | [ ] OK |

### 3.4 Network Infrastructure
| Parameter | Requirement | Available | Status |
|-----------|-------------|-----------|--------|
| **Network Type** | Ethernet 100/1000 Mbps | _____ Mbps | [ ] OK |
| **Network Jack** | RJ45 available | [ ] Available | [ ] OK |
| **IP Address Range** | Static IP available | ___.___.___.___  | [ ] OK |
| **Firewall Rules** | UDP 9001 open (lights) | [ ] Configured | [ ] OK |
| **File Server Access** | SMB/CIFS (optional) | [ ] Configured | [ ] OK |
| **Internet Access** | Not required | [ ] N/A | [ ] OK |

### 3.5 Physical Installation
**Dimensions Verified**:
- System footprint: _____ mm (W) × _____ mm (D) × _____ mm (H)
- Clearance around system: _____ mm (all sides)
- Access to front panel: [ ] OK
- Access to emergency stop: [ ] OK
- Mounting: [ ] Floor [ ] Table [ ] Cart
- Stability: [ ] Verified (no wobble)

**Ergonomics**:
- Operator standing position: [ ] Comfortable
- Monitor height: [ ] Appropriate
- Part loading height: [ ] Appropriate
- Safety clearances: [ ] Adequate

---

## 4. Pre-SAT Checklist

### 4.1 Shipping and Receiving
- [ ] System received on expected date
- [ ] Packaging inspected for damage
- [ ] All boxes/crates accounted for (qty: _____)
- [ ] Unpacking performed carefully
- [ ] All components present (verified against packing list)
- [ ] No visible shipping damage
- [ ] Packaging materials disposed per site policy

### 4.2 Physical Installation
- [ ] System moved to installation location
- [ ] Positioned per layout plan
- [ ] Leveled (turntable level verified)
- [ ] Secured to floor/table (if required)
- [ ] Adequate clearance on all sides
- [ ] Emergency stop accessible
- [ ] Lighting adequate for operator

### 4.3 Electrical Connection
- [ ] Power requirements verified (voltage, amperage)
- [ ] Dedicated circuit breaker installed
- [ ] Power cable connected
- [ ] Ground connection verified
- [ ] Surge protector installed (if applicable)
- [ ] Emergency stop tested
- [ ] Power-on test successful

### 4.4 Network Connection
- [ ] Network cable run to system location
- [ ] Cable tested for continuity
- [ ] Connected to system Ethernet port
- [ ] IP address assigned (static or DHCP)
- [ ] Ping test successful
- [ ] LED controller reachable (if applicable)
- [ ] File server accessible (if applicable)

### 4.5 Hardware Verification Post-Shipping
- [ ] Cameras visually inspected (no damage)
- [ ] Camera connections verified (USB cables secure)
- [ ] Turntable manually rotated (no binding)
- [ ] Linear axis manually moved (no binding)
- [ ] All cables secure and properly routed
- [ ] No loose hardware or parts
- [ ] Protective covers/guards in place

### 4.6 Software and Data
- [ ] System powered on successfully
- [ ] Windows boots normally
- [ ] Application launches without errors
- [ ] Configuration file present (`user_settings.json`)
- [ ] AI models loaded successfully
- [ ] FAT calibration data imported (if applicable)
- [ ] Storage path configured (`captures/` directory)
- [ ] Network drives mapped (if applicable)

### 4.7 Customer Preparation
- [ ] Operators identified and scheduled for training
- [ ] Production parts available for testing (qty: _____)
- [ ] Golden sample(s) identified
- [ ] Known defect sample(s) identified (if available)
- [ ] Part ID naming convention established
- [ ] Quality criteria documented
- [ ] Production schedule accommodates SAT testing

---

## 5. SAT Test Procedures

---

## TEST SECTION 1: POST-INSTALLATION VERIFICATION

### SAT-POST-001: System Power-Up After Shipping
**Objective**: Verify system powers on correctly after shipping
**Acceptance Criteria**: System boots normally, no hardware faults

**Procedure**:
1. Verify power connections secure
2. Power on system
3. Observe boot sequence
4. Check for any error messages
5. Verify GPU detected (Device Manager)
6. Note boot time: _____ seconds

**Results**:
- [ ] Pass - System boots normally
- [ ] Fail - Issue: _______________________

**Notes**: _____________________________________________

---

### SAT-POST-002: Camera Function After Shipping
**Objective**: Verify cameras survived shipping intact
**Acceptance Criteria**: Both cameras connect and capture clear images

**Procedure**:
1. Launch Envista-Solomon application
2. Refresh and connect top camera
3. Capture test image from top camera
4. Inspect image quality:
   - [ ] No dead pixels
   - [ ] Focus sharp
   - [ ] No damage artifacts
5. Repeat for front camera

**Results**:
- Top Camera: [ ] Pass [ ] Fail: _____________
- Front Camera: [ ] Pass [ ] Fail: _____________
- [ ] Pass - Cameras functional
- [ ] Fail - Damage found: _______________________

**Notes**: _____________________________________________

---

### SAT-POST-003: Motion System Function After Shipping
**Objective**: Verify turntable and axis functional after shipping
**Acceptance Criteria**: No mechanical damage, smooth operation

**Procedure**:
1. Connect turntable
2. Home turntable
3. Perform 360° rotation
4. Verify smooth operation: [ ] Yes
5. Connect linear axis
6. Calibrate axis
7. Move full travel (0-100mm)
8. Verify smooth operation: [ ] Yes

**Results**:
- Turntable: [ ] Pass [ ] Fail: _____________
- Linear Axis: [ ] Pass [ ] Fail: _____________
- [ ] Pass - Motion systems functional
- [ ] Fail - Damage found: _______________________

**Notes**: _____________________________________________

---

### SAT-POST-004: Calibration Verification
**Objective**: Verify FAT calibration values still valid after shipping
**Acceptance Criteria**: Calibration within tolerances (±10% of FAT values)

**Procedure**:
1. Review FAT calibration data
2. Perform calibration checks:
   - Camera alignment: _____ mm offset (FAT: _____ mm)
   - Turntable home repeatability: ±_____ ° (FAT: ±_____ °)
   - Linear axis accuracy: ±_____ mm (FAT: ±_____ mm)
3. Re-calibrate if necessary

**Results**:
- Calibration Valid: [ ] Yes [ ] Re-calibration needed
- [ ] Pass - Calibration acceptable
- [ ] Fail - Requires re-calibration

**Re-Calibration Performed**: [ ] Yes [ ] No
**New Calibration Values Recorded**: [ ] Yes [ ] N/A

**Notes**: _____________________________________________

---

## TEST SECTION 2: SITE INTEGRATION

### SAT-SITE-001: Network Integration
**Objective**: Verify network connectivity in production environment
**Acceptance Criteria**: All network functions operational

**Procedure**:
1. Verify system on customer network
2. Ping test to LED controller:
   - IP: ___.___.___.___
   - Ping result: [ ] Success (time: _____ ms)
3. Test file server access (if applicable):
   - Path: _______________________
   - Access: [ ] Success
4. Verify no network conflicts
5. Test network during inspection (no dropouts)

**Results**:
- Network Connectivity: [ ] OK
- File Access: [ ] OK [ ] N/A
- LED Controller: [ ] Reachable
- [ ] Pass - Network integration successful
- [ ] Fail - Issue: _______________________

**Notes**: _____________________________________________

---

### SAT-SITE-002: Power Quality
**Objective**: Verify site power quality adequate
**Acceptance Criteria**: Voltage stable, no brownouts during operation

**Procedure**:
1. Measure voltage at system power inlet:
   - No load: _____ V
   - System running: _____ V
   - During inspection: _____ V
2. Verify voltage variation <±5%
3. Run inspection, monitor for power issues
4. Check for ground loops or noise

**Results**:
- Voltage Stability: [ ] OK
- No Power Interruptions: [ ] OK
- [ ] Pass - Power quality adequate
- [ ] Fail - Issue: _______________________

**Notes**: _____________________________________________

---

### SAT-SITE-003: Environmental Conditions
**Objective**: Verify production environment suitable
**Acceptance Criteria**: Temperature, lighting, vibration within specs

**Procedure**:
1. Measure ambient temperature: _____ °C
2. Measure ambient light at system: _____ lux
3. Observe vibration from nearby equipment
4. Run inspection, monitor for environmental issues:
   - Image quality affected by vibration: [ ] No
   - Ambient light causing glare: [ ] No
   - Temperature comfortable for continuous operation: [ ] Yes

**Results**:
- Temperature: [ ] OK (15-30°C)
- Lighting: [ ] OK (<500 lux ambient)
- Vibration: [ ] OK (<0.5g)
- [ ] Pass - Environment suitable
- [ ] Fail - Issue: _______________________

**Notes**: _____________________________________________

---

### SAT-SITE-004: Storage Path Configuration
**Objective**: Verify data storage configured for site
**Acceptance Criteria**: Images save to correct location, adequate space

**Procedure**:
1. Verify storage path: _______________________
2. Check available disk space: _____ GB free
3. Estimate storage per day: _____ parts × _____ MB/part = _____ GB/day
4. Verify storage adequate for _____ days
5. Run test inspection, verify files save correctly
6. Check file permissions (read/write)

**Results**:
- Storage Path: [ ] Configured
- Disk Space: [ ] Adequate (>100GB recommended)
- File Access: [ ] OK
- [ ] Pass - Storage configured correctly
- [ ] Fail - Issue: _______________________

**Notes**: _____________________________________________

---

## TEST SECTION 3: CUSTOMER PART VALIDATION

### SAT-PART-001: First Customer Part Inspection
**Objective**: Run first inspection with actual customer part
**Acceptance Criteria**: Inspection completes successfully

**Procedure**:
1. Obtain first customer part
2. Part Description: _______________________
3. Expected attachments: _____
4. Place part on turntable
5. Enter Part ID: "_______________________"
6. Run inspection
7. Record results:
   - Step 1 detections: _____
   - Step 2 rotations: _____
   - Step 3 front detections: _____
   - Step 4 defects: _____
   - Cycle time: _____ seconds
   - Overall result: [ ] PASS [ ] FAIL

**Results**:
- Inspection Completed: [ ] Yes
- Results Reasonable: [ ] Yes
- [ ] Pass - First customer part successful
- [ ] Fail - Issue: _______________________

**Saved Data Path**: _______________________________________

**Notes**: _____________________________________________

---

### SAT-PART-002: Golden Sample Validation
**Objective**: Verify system detects customer's golden sample correctly
**Acceptance Criteria**:
- All expected attachments detected
- No false positives
- No defects detected (PASS result)

**Procedure**:
1. Obtain customer's golden sample (verified good part)
2. Golden Sample ID: _______________________
3. Expected attachments: _____
4. Run inspection
5. Review results:
   - Detections match expected: [ ] Yes
   - False positives: _____ (should be 0)
   - Defects detected: _____ (should be 0)
   - Overall result: [ ] PASS (expected)
6. Review annotated images with customer
7. Customer confirms detection correct: [ ] Yes

**Results**:
- Detection Accuracy: _____ / _____ (100% expected)
- Customer Approval: [ ] Yes
- [ ] Pass - Golden sample detected correctly
- [ ] Fail - Issue: _______________________

**Notes**: _____________________________________________

---

### SAT-PART-003: Known Defect Sample Validation
**Objective**: Verify system detects customer's known defects
**Acceptance Criteria**: Known defects detected, overall result FAIL

**Procedure**:
1. Obtain customer's known defect sample
2. Defect Sample ID: _______________________
3. Known defects: _______________________
4. Run inspection
5. Review results:
   - Defects detected: _____
   - Defect types match known: [ ] Yes
   - Confidence scores: _______________________
   - Overall result: [ ] FAIL (expected)
6. Review defect annotations with customer
7. Customer confirms defect detection correct: [ ] Yes

**Results**:
- Defect Detection: [ ] Correct
- Customer Approval: [ ] Yes
- [ ] Pass - Defects detected correctly
- [ ] Fail - Issue: _______________________

**Notes**: _____________________________________________

---

### SAT-PART-004: Part Variation Testing
**Objective**: Verify system handles normal part variation
**Acceptance Criteria**: Consistent results across multiple samples

**Procedure**:
1. Obtain 5-10 production parts (mix of good/bad if available)
2. Inspect each part
3. Record results:

| Part ID | Detections | Defects | Result | Cycle Time | Notes |
|---------|------------|---------|--------|------------|-------|
| 1. | | | | | |
| 2. | | | | | |
| 3. | | | | | |
| 4. | | | | | |
| 5. | | | | | |
| 6. | | | | | |
| 7. | | | | | |
| 8. | | | | | |
| 9. | | | | | |
| 10. | | | | | |

4. Calculate statistics:
   - Average cycle time: _____ seconds
   - Cycle time range: _____ - _____ seconds
   - Consistent detection count: [ ] Yes
5. Review results with customer

**Results**:
- Consistency: [ ] Good
- Customer Approval: [ ] Yes
- [ ] Pass - Handles part variation
- [ ] Fail - Issue: _______________________

**Notes**: _____________________________________________

---

### SAT-PART-005: Cycle Time Validation
**Objective**: Verify cycle time meets production requirements
**Acceptance Criteria**: Cycle time ≤ target time for customer parts

**Procedure**:
1. Customer's target cycle time: _____ seconds
2. Run 10 consecutive inspections
3. Record cycle times:

| Run | Cycle Time (s) | Within Target? |
|-----|----------------|----------------|
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

4. Calculate average: _____ seconds
5. Compare to target: [ ] Meets requirement

**Results**:
- Average Cycle Time: _____ seconds
- Target Cycle Time: _____ seconds
- [ ] Pass - Cycle time acceptable
- [ ] Fail - Too slow by _____ seconds

**Notes**: _____________________________________________

---

### SAT-PART-006: Defect Detection Sensitivity
**Objective**: Verify defect detection meets customer's quality standards
**Acceptance Criteria**: Sensitivity acceptable to customer QA team

**Procedure**:
1. With customer QA engineer, review defect threshold
2. Current threshold: _____
3. Test with borderline defect samples:
   - Sample 1: Customer expects [ ] PASS [ ] FAIL → System result: [ ] PASS [ ] FAIL
   - Sample 2: Customer expects [ ] PASS [ ] FAIL → System result: [ ] PASS [ ] FAIL
   - Sample 3: Customer expects [ ] PASS [ ] FAIL → System result: [ ] PASS [ ] FAIL
4. Adjust threshold if needed: New threshold: _____
5. Retest after adjustment
6. Customer QA approval: [ ] Yes

**Results**:
- Threshold: _____ (final)
- Agreement with Customer QA: [ ] Yes
- [ ] Pass - Sensitivity appropriate
- [ ] Fail - Cannot meet requirements: _______________________

**Notes**: _____________________________________________

---

## TEST SECTION 4: PRODUCTION WORKFLOW INTEGRATION

### SAT-WORK-001: Part Loading/Unloading
**Objective**: Verify part loading ergonomics and safety
**Acceptance Criteria**: Operator can load/unload safely and comfortably

**Procedure**:
1. Operator loads part on turntable
2. Observe process:
   - Height comfortable: [ ] Yes
   - Can reach turntable easily: [ ] Yes
   - No awkward posture: [ ] Yes
   - Part placement repeatable: [ ] Yes
3. Operator unloads part
4. Repeat 10 times
5. Operator feedback on ergonomics: _______________________

**Results**:
- Ergonomics: [ ] Acceptable
- Safety: [ ] No pinch points or hazards
- [ ] Pass - Loading/unloading acceptable
- [ ] Fail - Issue: _______________________

**Notes**: _____________________________________________

---

### SAT-WORK-002: Part ID Entry Workflow
**Objective**: Verify Part ID entry fits customer workflow
**Acceptance Criteria**: Part ID entry fast and error-free

**Procedure**:
1. Determine Part ID method:
   - [ ] Manual keyboard entry
   - [ ] Barcode scanner (if available)
   - [ ] Auto-increment (counter)
   - [ ] Other: _______________________
2. Test Part ID entry 10 times
3. Time average entry: _____ seconds
4. Errors encountered: _____
5. Operator feedback: _______________________

**Results**:
- Entry Time: [ ] Acceptable
- Error Rate: [ ] Low
- [ ] Pass - Part ID workflow acceptable
- [ ] Fail - Issue: _______________________

**Recommendations**: _____________________________________________

**Notes**: _____________________________________________

---

### SAT-WORK-003: Results Review Workflow
**Objective**: Verify operator can review results effectively
**Acceptance Criteria**: Operator understands results and can make pass/fail decision

**Procedure**:
1. Run inspection
2. Operator reviews results:
   - Detection table visible: [ ] Yes
   - Defect ledger visible: [ ] Yes
   - Images clear and useful: [ ] Yes
   - Overall result clear (PASS/FAIL): [ ] Yes
3. Operator identifies which attachment has defect: [ ] Correct
4. Operator can navigate to saved images: [ ] Yes
5. Time to review results: _____ seconds

**Results**:
- Results Clarity: [ ] Good
- Operator Understanding: [ ] Yes
- [ ] Pass - Results review workflow acceptable
- [ ] Fail - Issue: _______________________

**Notes**: _____________________________________________

---

### SAT-WORK-004: Failed Part Handling
**Objective**: Verify process for handling failed parts
**Acceptance Criteria**: Failed part process clear and documented

**Procedure**:
1. Run inspection with defect sample (intentional fail)
2. System shows FAIL result: [ ] Yes
3. Establish failed part procedure:
   - [ ] Mark part physically (tag/sticker)
   - [ ] Place in reject bin
   - [ ] Log in quality tracking system
   - [ ] Notify supervisor
   - [ ] Other: _______________________
4. Operator performs procedure
5. Procedure documented: [ ] Yes

**Results**:
- Procedure Clear: [ ] Yes
- Operator Trained: [ ] Yes
- [ ] Pass - Failed part handling defined
- [ ] Fail - Needs clarification

**Documented Procedure**: _____________________________________________
_____________________________________________

**Notes**: _____________________________________________

---

### SAT-WORK-005: Data Management Workflow
**Objective**: Verify data saved and accessible per customer requirements
**Acceptance Criteria**: Data organized, accessible, and meeting retention policy

**Procedure**:
1. Review data storage structure
2. Verify images saved: [ ] Yes
3. Verify cycle time logged: [ ] Yes
4. Check data accessibility:
   - Operators can view past results: [ ] Yes [ ] Not Required
   - QA team can access archives: [ ] Yes [ ] Not Required
   - Data backed up: [ ] Yes [ ] Customer's responsibility
5. Verify retention policy: Keep data for _____ days/months

**Results**:
- Data Organization: [ ] Acceptable
- Accessibility: [ ] Meets requirements
- [ ] Pass - Data management acceptable
- [ ] Fail - Issue: _______________________

**Notes**: _____________________________________________

---

### SAT-WORK-006: Shift Changeover
**Objective**: Verify system state clear during shift changes
**Acceptance Criteria**: Incoming operator understands system status

**Procedure**:
1. Simulate shift changeover
2. Outgoing operator notes:
   - Current Part ID: _______________________
   - Parts inspected this shift: _____
   - Any issues encountered: _______________________
   - System state: [ ] Ready [ ] Error [ ] Maintenance needed
3. Incoming operator reviews system
4. Incoming operator runs test inspection
5. Transition smooth: [ ] Yes

**Results**:
- Handover Process: [ ] Clear
- System State Visible: [ ] Yes
- [ ] Pass - Shift changeover acceptable
- [ ] Fail - Issue: _______________________

**Notes**: _____________________________________________

---

## TEST SECTION 5: ERROR HANDLING & RECOVERY

### SAT-ERROR-001: Error Recovery Training
**Objective**: Verify operators can recover from common errors
**Acceptance Criteria**: Operators trained on error recovery procedures

**Procedure**:
Test operator response to simulated errors:
1. **Camera Disconnection**:
   - Simulate: Disconnect camera USB
   - Operator action: _______________________
   - Recovery time: _____ minutes
   - Successful: [ ] Yes
2. **Motion Fault**:
   - Simulate: Obstruct turntable
   - Operator action: _______________________
   - Recovery time: _____ minutes
   - Successful: [ ] Yes
3. **Part Misalignment**:
   - Simulate: Part off-center
   - Operator action: _______________________
   - Recovery time: _____ minutes
   - Successful: [ ] Yes

**Results**:
- Operator Competency: [ ] Good
- Procedures Clear: [ ] Yes
- [ ] Pass - Operators can recover from errors
- [ ] Fail - Additional training needed

**Notes**: _____________________________________________

---

### SAT-ERROR-002: Emergency Stop Procedure
**Objective**: Verify operators understand emergency stop
**Acceptance Criteria**: All operators know when and how to use E-stop

**Procedure**:
1. Show operator emergency stop location
2. Explain when to use:
   - [ ] Unsafe condition
   - [ ] Mechanical binding
   - [ ] Person in danger
   - [ ] Other emergency
3. Test procedure:
   - Operator presses E-stop: [ ] Correct
   - Motion stops immediately: [ ] Yes
   - Operator can reset: [ ] Yes
4. Document in operator checklist

**Results**:
- Operator Understanding: [ ] Yes
- E-stop Functional: [ ] Yes
- [ ] Pass - Emergency stop procedure clear
- [ ] Fail - Needs clarification

**Notes**: _____________________________________________

---

### SAT-ERROR-003: Maintenance Contact Procedure
**Objective**: Verify operators know how to get support
**Acceptance Criteria**: Contact info clear, escalation path defined

**Procedure**:
1. Provide contact information:
   - **Level 1 - Internal**: _______________________
   - **Level 2 - Manufacturer Support**: _______________________
   - **Email**: _______________________
   - **Phone**: _______________________
   - **Response Time**: _____ hours (normal), _____ hours (emergency)
2. Establish escalation:
   - Try to resolve internally: _____ minutes
   - Contact manufacturer: If not resolved
3. Operator has contact info: [ ] Yes

**Results**:
- Contact Info Documented: [ ] Yes
- Escalation Clear: [ ] Yes
- [ ] Pass - Support procedure defined
- [ ] Fail - Needs clarification

**Contact Sheet Posted Near System**: [ ] Yes

**Notes**: _____________________________________________

---

## 6. Site Integration Testing

### 6.1 Production Line Integration
**Objective**: Verify system fits into production workflow
**Pass/Fail**: [ ] Pass [ ] Fail

**Integration Points Tested**:
- [ ] Upstream process (part delivery to inspection)
- [ ] Inspection station operation
- [ ] Downstream process (part removal to next station)
- [ ] Quality data logging (if integrated)
- [ ] Production tracking (if integrated)

**Production Flow Test**:
1. Simulate production scenario
2. Inspect _____ consecutive parts
3. Monitor:
   - Cycle time consistent: [ ] Yes
   - No bottlenecks: [ ] Yes
   - Operators comfortable: [ ] Yes
   - Results actionable: [ ] Yes

**Integration Issues Identified**: _____________________________________________
_____________________________________________

**Resolution**: _____________________________________________

---

### 6.2 Environmental Stress Test
**Objective**: Verify system stable over extended operation
**Pass/Fail**: [ ] Pass [ ] Fail

**Test Duration**: _____ hours (minimum 4 hours recommended)

**Procedure**:
1. Run continuous inspections
2. Monitor system:
   - Temperature drift: _____ °C (should be stable)
   - No performance degradation: [ ] Yes
   - No errors or crashes: [ ] Yes
   - Image quality consistent: [ ] Yes

**Parts Inspected**: _____
**Total Duration**: _____ hours
**Issues Encountered**: _____________________________________________

**Results**:
- [ ] Pass - Stable extended operation
- [ ] Fail - Issue: _______________________

---

## 7. Operator Training Verification

### 7.1 Training Completion Checklist

**Training Sessions Completed**:
- [ ] System Overview (1 hour)
- [ ] Basic Operation (2 hours, hands-on)
- [ ] Results Interpretation (1 hour)
- [ ] Error Recovery (1 hour, hands-on)
- [ ] Maintenance & Cleaning (30 min)
- [ ] Safety Procedures (30 min)

**Total Training Time**: _____ hours

### 7.2 Operator Competency Assessment

For each operator, verify competency:

#### Operator 1: _______________________

| Task | Can Perform Independently | Supervisor Sign-off |
|------|---------------------------|---------------------|
| Power on/off system | [ ] Yes | |
| Connect cameras | [ ] Yes | |
| Load AI models | [ ] Yes | |
| Home turntable and axis | [ ] Yes | |
| Load part on turntable | [ ] Yes | |
| Enter Part ID | [ ] Yes | |
| Start inspection | [ ] Yes | |
| Interpret results (PASS/FAIL) | [ ] Yes | |
| Save/review images | [ ] Yes | |
| Handle failed part | [ ] Yes | |
| Recover from camera error | [ ] Yes | |
| Recover from motion error | [ ] Yes | |
| Use emergency stop | [ ] Yes | |
| Contact support | [ ] Yes | |

**Overall Competency**: [ ] Qualified [ ] Needs Additional Training

**Operator Signature**: _____________________________ **Date**: __________

---

#### Operator 2: _______________________

| Task | Can Perform Independently | Supervisor Sign-off |
|------|---------------------------|---------------------|
| Power on/off system | [ ] Yes | |
| Connect cameras | [ ] Yes | |
| Load AI models | [ ] Yes | |
| Home turntable and axis | [ ] Yes | |
| Load part on turntable | [ ] Yes | |
| Enter Part ID | [ ] Yes | |
| Start inspection | [ ] Yes | |
| Interpret results (PASS/FAIL) | [ ] Yes | |
| Save/review images | [ ] Yes | |
| Handle failed part | [ ] Yes | |
| Recover from camera error | [ ] Yes | |
| Recover from motion error | [ ] Yes | |
| Use emergency stop | [ ] Yes | |
| Contact support | [ ] Yes | |

**Overall Competency**: [ ] Qualified [ ] Needs Additional Training

**Operator Signature**: _____________________________ **Date**: __________

---

### 7.3 Training Materials Provided

- [ ] User Manual (printed/PDF)
- [ ] Quick Start Guide (laminated, posted near system)
- [ ] Troubleshooting Guide
- [ ] Safety Instructions
- [ ] Maintenance Schedule
- [ ] Contact Information Sheet
- [ ] Training Video (optional, if available)

**Materials Location**: _____________________________________________

---

## 8. Production Readiness

### 8.1 Final Configuration

**System Configuration Finalized**:
- [ ] Camera positions locked
- [ ] Turntable home position marked
- [ ] Linear axis home position set
- [ ] Light intensities optimized
- [ ] Model paths configured
- [ ] Storage path configured
- [ ] Defect threshold set: _____
- [ ] Crop size set: _____ pixels
- [ ] Part ID format agreed: _______________________

**Configuration Backup**:
- [ ] `user_settings.json` backed up to: _______________________
- [ ] Calibration images saved to: _______________________

---

### 8.2 Maintenance Schedule

**Daily Maintenance** (Operator):
- [ ] Clean camera lenses
- [ ] Check for loose cables
- [ ] Verify emergency stop functional
- [ ] Clear turntable of debris

**Weekly Maintenance** (Operator/Technician):
- [ ] Check camera alignment
- [ ] Verify motion smoothness
- [ ] Clean turntable surface
- [ ] Check storage space

**Monthly Maintenance** (Technician):
- [ ] Verify calibration
- [ ] Check all cable connections
- [ ] Inspect for wear
- [ ] Software updates (if available)

**Quarterly Maintenance** (Technician/Manufacturer):
- [ ] Full calibration check
- [ ] Performance benchmark
- [ ] Preventive maintenance
- [ ] Model updates (if needed)

**Maintenance Schedule Posted**: [ ] Yes
**Location**: _____________________________________________

---

### 8.3 Production Approval Checklist

Before releasing to production, verify:

**Hardware**:
- [ ] All hardware functional
- [ ] Cameras aligned and calibrated
- [ ] Motion systems accurate
- [ ] Emergency stop tested
- [ ] No loose or damaged parts

**Software**:
- [ ] Application loads without errors
- [ ] All AI models loaded
- [ ] Configuration finalized
- [ ] Storage path accessible
- [ ] Network connectivity verified

**Integration**:
- [ ] Fits production workflow
- [ ] Cycle time meets requirements
- [ ] Data management acceptable
- [ ] Shift changeover clear

**Training**:
- [ ] All operators trained
- [ ] Competency verified
- [ ] Training materials provided
- [ ] Support contacts documented

**Validation**:
- [ ] Golden sample validated
- [ ] Defect detection validated
- [ ] Part variation tested
- [ ] Extended run completed

**Customer Approval**:
- [ ] Customer QA approval
- [ ] Production manager approval
- [ ] Safety approval

---

## 9. SAT Results Summary

### 9.1 Test Summary by Section

| Test Section | Total Tests | Passed | Failed | N/A | Pass Rate |
|--------------|-------------|--------|--------|-----|-----------|
| Post-Installation | 4 | ___ | ___ | ___ | ___% |
| Site Integration | 4 | ___ | ___ | ___ | ___% |
| Customer Part Validation | 6 | ___ | ___ | ___ | ___% |
| Workflow Integration | 6 | ___ | ___ | ___ | ___% |
| Error Handling | 3 | ___ | ___ | ___ | ___% |
| **TOTAL** | **27** | **___** | **___** | **___** | **___%** |

### 9.2 Performance Summary

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Cycle Time (avg) | _____ s | _____ s | [ ] Pass |
| Detection Accuracy | >95% | _____ % | [ ] Pass |
| Defect Detection | >90% | _____ % | [ ] Pass |
| System Uptime (test period) | >95% | _____ % | [ ] Pass |
| Operator Satisfaction | Good | [ ] Good [ ] Fair [ ] Poor | [ ] Pass |

### 9.3 Issues Identified

#### Critical Issues (Must Resolve Before Production)
| Issue ID | Description | Severity | Status | Resolution |
|----------|-------------|----------|--------|------------|
| | | | | |

#### Minor Issues (Can Be Resolved During Production)
| Issue ID | Description | Severity | Status | Notes |
|----------|-------------|----------|--------|-------|
| | | | | |

### 9.4 Site-Specific Recommendations

**Operational Recommendations**:
1. _____________________________________________
2. _____________________________________________
3. _____________________________________________

**Future Enhancements**:
1. _____________________________________________
2. _____________________________________________
3. _____________________________________________

**Customer Action Items**:
1. _____________________________________________
2. _____________________________________________

---

## 10. Sign-off & Handover

### 10.1 SAT Completion

**SAT Start Date**: _______________________
**SAT End Date**: _______________________
**Total Duration**: _______ days
**Installation Team On-Site**: _______ days

### 10.2 Installation Team Sign-off

I certify that the Envista-Solomon system has been installed, configured, and tested according to this SAT document. The system is ready for production use subject to customer acceptance.

**Field Service Engineer**:
- Name: _______________________________
- Company: _____________________________
- Signature: ___________________________
- Date: _______________________________

**Software Support**:
- Name: _______________________________
- Signature: ___________________________
- Date: _______________________________

**Training Specialist**:
- Name: _______________________________
- Signature: ___________________________
- Date: _______________________________

### 10.3 Customer Acceptance Sign-off

I certify that the Envista-Solomon system has been installed at our facility and tested with our production parts. The system meets our requirements and is approved for production use.

**Production Manager**:
- Name: _______________________________
- Company: _____________________________
- Signature: ___________________________
- Date: _______________________________

**Quality Engineer**:
- Name: _______________________________
- Signature: ___________________________
- Date: _______________________________

**Operator(s) - Trained and Qualified**:
- Name: _______________________________
- Signature: ___________________________
- Date: _______________________________

- Name: _______________________________
- Signature: ___________________________
- Date: _______________________________

### 10.4 SAT Result

- [ ] **PASS** - System approved for production use
- [ ] **CONDITIONAL PASS** - System approved with action items (see Section 9.3)
- [ ] **FAIL** - System requires additional work before production use

**Conditions/Action Items** (if applicable):
_____________________________________________
_____________________________________________
_____________________________________________

### 10.5 Handover to Production

**Production Start Date**: _______________________

**System Ownership Transfer**:
- System physically handed over to: _______________________
- Primary point of contact: _______________________
- Phone: _______________________
- Email: _______________________

**Support Coverage**:
- Warranty Period: _______ months from SAT completion
- Support Hotline: _______________________
- Remote Support: [ ] Available [ ] Not Available
- On-site Support: [ ] Available (response time: _____ hours)

**Spare Parts Provided**:
- [ ] USB cables (qty: _____)
- [ ] Serial cables (qty: _____)
- [ ] Calibration target
- [ ] Other: _______________________

**Documentation Package Delivered**:
- [ ] User Manual
- [ ] FAT Document (copy)
- [ ] SAT Document (this document)
- [ ] Training Materials
- [ ] Maintenance Schedule
- [ ] Contact Information
- [ ] Software License Info (if applicable)

**Post-Installation Support Plan**:
- Week 1: Daily check-in (remote)
- Week 2-4: Twice weekly check-in (remote)
- Month 2-3: Weekly check-in (remote)
- Month 4-12: Monthly check-in (remote)
- As-needed: On-site support (response time: _____ hours)

---

## 11. Appendices

### Appendix A: Site Installation Photos

Photos documenting installation:
- [ ] Overall system view
- [ ] Camera positions
- [ ] Turntable setup
- [ ] Linear axis setup
- [ ] Operator workstation
- [ ] Network connections
- [ ] Power connections
- [ ] Emergency stop
- [ ] Safety labels/signage

**Photo Archive Location**: _____________________________________________

---

### Appendix B: Customer Part Images

Sample images from SAT testing:
- [ ] Golden sample (step-01 through step-04)
- [ ] Defect sample (step-01 through step-04)
- [ ] Part variation samples
- [ ] Edge cases (if encountered)

**Image Archive Location**: _____________________________________________

---

### Appendix C: Training Attendance Records

| Operator Name | Training Date | Duration (hrs) | Signature |
|---------------|---------------|----------------|-----------|
| | | | |
| | | | |
| | | | |
| | | | |

---

### Appendix D: Site-Specific Configuration

**Customer-Specific Settings**:
- Part ID Prefix: _______________________
- Defect Threshold: _____
- Crop Size: _____ pixels
- Light Intensity Top: _____ mA
- Light Intensity Front: _____ mA
- Network IP: ___.___.___.___
- Storage Path: _______________________

**Custom Workflows**:
_____________________________________________
_____________________________________________

**Special Requirements**:
_____________________________________________
_____________________________________________

---

### Appendix E: Known Limitations

Document any known limitations specific to customer's application:

| Limitation | Impact | Workaround |
|------------|--------|------------|
| | | |
| | | |

---

### Appendix F: Future Enhancement Requests

Customer requests for future consideration:

| Request | Priority | Notes |
|---------|----------|-------|
| | [ ] High [ ] Medium [ ] Low | |
| | [ ] High [ ] Medium [ ] Low | |
| | [ ] High [ ] Medium [ ] Low | |

---

**END OF SAT DOCUMENT**

**Document Control**:
- Document ID: SAT-EnvistaSolomon-[CustomerName]-001
- Version: 1.0
- Created By: _______________________________
- Reviewed By: _______________________________
- Approved By: _______________________________
- Date: _______________________________

---

**Customer Copy**: This document becomes customer property upon SAT completion.
**Manufacturer Copy**: Retain for warranty and support purposes.
