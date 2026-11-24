# FYP Notes - Progress Summary

## Step-by-Step Progress

### 1 - Project Setup

- Created the project structure and Python environment.
- Organized folders for datasets, detector, roles, depth, perturbations, metrics, experiments, and results.

### 2 - KINS Loader

- Implemented loading of KINS annotations.
- Added helpers for images, annotations, categories, and occlusion ratio calculation.

### 3 - Target Filtering

- Filtered suitable partially occluded vehicle targets.
- Focused on useful visible size and moderate occlusion.

### 4 - Sample Visualization

- Visualized targets, visible regions, amodal regions, and occluders.
- Used this to verify KINS annotation interpretation.

### 5 - YOLO Detector Wrapper

- Added a YOLO11n wrapper.
- Returned class, confidence, and bounding box for each detection.

### 6 - Target Matching

- Matched KINS targets to YOLO detections using class consistency and bbox IoU.
- Used a minimum IoU threshold of about 0.30.

### 7 - Scene Role Construction

- Built target, foreground occluder, and context masks.
- Used KINS annotations for the original role construction.

### 8 - Occluder Selection Refinement

- Improved foreground occluder selection.
- Used visible/hidden target geometry and occlusion relationships.

### 9 - Perturbation Functions

- Implemented black masking and Gaussian blur.

### 10 - Detection Metrics

- Added confidence change, bbox IoU comparison, and target survival/disappearance tracking.

### 11 - Single-Target Experiment

- Ran the complete perturbation workflow on one target.
- Tested target, occluder, and context using mask and blur.

### 12 - Structured CSV Output

- Saved perturbation results in CSV format.

### 13 - Batch Experiment

- Extended the experiment to multiple targets.

### 14 - Manual Validation

- Manually checked selected examples.
- Removed unsuitable or incorrect cases.

### 15 - Final Validated Set

- Finalized a set of 40 validated targets.

### 16 - Aggregate Role Analysis

- Compared target, occluder, and context perturbations across all 40 targets.
- Target masking produced the strongest average effect.
- Context also showed meaningful influence.
- Occluder perturbation was weaker and more mixed.

### 17 - Per-Target Analysis

- Examined role effects target-by-target.
- Confirmed that role influence varies by scene.

### 18 - Random Mask Baseline

- Compared semantic target masking with random local masking.
- Target masking was stronger on average than random masking.

### 19 - Synthetic Occlusion

- Added controlled target occlusion at increasing levels.
- Observed monotonic degradation in confidence, localization, and survival.

### 20 - D-RISE Comparison

- Added a simplified D-RISE baseline.
- D-RISE usually ranked the target as most salient.
- Role perturbation and D-RISE showed partial agreement but measure different things.

---

## 21 - Depth Anything V2 Integration

- Integrated **Depth Anything V2 Small**.
- Generated monocular relative depth maps.
- Measured median depth inside target and occluder masks.
- In the current output convention, larger raw depth value means closer object.
- Relative depth ordering matched the expected KINS front/back relationship for all 40 selected validated pairs.
- This showed monocular depth could potentially replace explicit KINS occlusion-order annotations.

---

## 22 - Automatic Role Inference Using Depth

- Created automatic foreground-occluder inference using KINS visible masks/boxes plus predicted monocular depth.
- Removed direct dependency on KINS `oco_id` and `ico_id`.
- Candidate occluders were selected using spatial overlap and positive depth difference.
- Batch result:
  - 38 / 40 correct
  - 95% role-inference accuracy
- When the correct physical occluder was inferred, downstream perturbation results matched the original KINS-role results exactly.
- Conclusion: predicted depth could recover occlusion relationships well when geometry was still supplied by KINS.

---

## 23 - Fully Predicted Role Inference

### 23A - YOLO11n-seg Integration

- Added YOLO11n-seg.
- Predicted instance masks, bounding boxes, classes, and confidence scores.

### 23B - Predicted Target Matching

- Matched the known evaluation target to a YOLO11n-seg instance.
- KINS was only used to identify the target during validation.
- Role inference no longer depended on KINS target/occluder geometry.

### 23C - Predicted Role Inference

- Built target, occluder, and context roles from predicted masks, boxes, and depth.
- Removed KINS geometry from occluder selection.

### 23D - Instance Deduplication

- Found duplicate YOLO segmentation predictions for the same physical object.
- Added class-independent deduplication using bbox IoU and mask IoU.
- Retained the higher-confidence instance.

### 23E - Physical Occluder Validation

- Compared predicted occluders with KINS reference occluders.
- Used physical geometry rather than exact class agreement.
- A prediction was considered correct when:
  - bbox IoU >= 0.50
  - OR mask IoU >= 0.50

### 23F - Initial Fully Predicted Batch Evaluation

- Initial result:
  - 36 correct
  - 2 wrong
  - 2 no prediction
  - 90% overall accuracy
- Main failure cases:
  - 1873
  - 4042
  - 5605
  - 7543

### 23G - Failure Analysis

- Investigated each failed target in detail.
- Identified:
  - segmentation/localization issues
  - depth-order issues
  - candidate geometry issues
  - ranking issues

### 23H - Expanded Target Region

- Expanded the predicted target box to include nearby candidate occluders.
- This helped but did not solve all cases.

### 23I - Distance-Aware Candidate Geometry

- Replaced strict visible-box overlap with a proximity-aware rule.
- Candidate acceptance became:
  - sufficient overlap
  - OR sufficiently small normalized spatial gap
- Kept positive depth difference as a foreground constraint.
- Final ranking used:
  - proximity
  - depth advantage
  - detection confidence
- Final normalized-gap threshold:
  - `0.36`
- This fixed previous failures such as:
  - target 5605
  - target 7543

### 23 - Final Result

- 37 / 40 correct physical occluders
- 2 wrong
- 1 no prediction
- 0 target segmentation match failures
- **92.5% overall accuracy**
- **94.87% accuracy when a prediction exists**
- Median predicted occluder bbox IoU:
  - **0.9317**
- Median predicted occluder mask IoU:
  - **0.7836**

---

# Main Findings So Far

- The **target region is usually the strongest contributor** to detector confidence.
- **Context can also have a noticeable effect**.
- **Foreground occluder influence is weak or mixed on average**.
- The project should not claim that occluders are always highly important.
- Increasing synthetic target occlusion causes:
  - lower confidence
  - lower survival
  - lower bbox stability
- Random masking has a weaker average effect than semantic target masking.
- D-RISE usually identifies the target as most salient.
- D-RISE and role perturbation measure different things:
  - D-RISE = saliency / evidence preservation
  - role perturbation = intervention-based influence
- Monocular depth is useful for recovering foreground/background order.
- Depth ordering matched all 40 selected validated target/occluder pairs.
- 22 automatic depth-based role inference achieved **95% accuracy**.
- 23 fully predicted role inference achieved **92.5% overall accuracy** and **94.87% accuracy when a prediction exists**.
- The fully predicted method shows that ground-truth occlusion annotations are not required for role inference.

---

# Important Methodological Changes Made

- Moved from **KINS annotation-defined roles** toward **predicted roles**.
- Removed dependency on KINS `oco_id` and `ico_id` in 22.
- Removed KINS mask/box dependency from role inference in 23.
- Added monocular depth as the main foreground/background cue.
- Added YOLO11n-seg to obtain predicted masks and boxes.
- Changed occluder evaluation from exact class matching to **physical object matching using geometry**.
- Added instance deduplication because YOLO11n-seg sometimes predicts the same physical object more than once.
- Replaced strict target/occluder overlap with **distance-aware geometry**.
- Added normalized bbox gap as a spatial cue.
- Changed candidate ranking to combine:
  - proximity
  - depth difference
  - confidence
- Avoided forcing candidate class to match target class, because real occluders can belong to any class.
- Froze the normalized-gap threshold at `0.36` after failure analysis to avoid continuous overfitting.
- Kept KINS only as an evaluation reference in 23.
- In a final real-image system, the target can be selected directly from detector predictions instead of KINS.
