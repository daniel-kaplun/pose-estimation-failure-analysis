# Experiment 2: Synthetic Occlusion Evaluation

This experiment evaluates HRNet-W32 on a synthetic dataset with perfect ground truth keypoints and systematically increasing occlusion. It is created to understand model limitations in controlled environments.

---

## Dataset Setup

Mixamo rig, barbell back squat animation. 70 frames, 15 iterations.  
Camera placed at an angle between left and front side, in such a way that whole rig is seen.  
The occluder is a flat square plane placed on the rig's left shoulder that expands in the direction the model faces. Scale increases with each iteration.  

**Total images:** 15 × 70 = 1050

---

## Rendering

- 3D joints projected from world to camera view 
- Icospheres attached to joints for perfect ground truth keypoints
- Ray casting for visibility of keypoints

---

## Model

- HRNet-W32  
- COCO pretrained  
- No fine tuning, evaluation only

---

## Evaluation Metrics

Average Precision (AP), as defined in this project, is the percentage of correctly predicted keypoints within 20 pixels of the ground truth. Used to measure all multi-keypoint graphs.  

For the left shoulder graphs, per-keypoint localization error was computed using Euclidean distance in pixel space.

---

## Results

- AP drops quickly after second iteration  
- Even with no occlusion, AP drops when model is squatted
- Stabilizes at high occlusion
- Shoulder error increases rapidly

---

## Limitations

- Single camera  
- Single motion  
- Gap between synthetic and real world  
- No training
