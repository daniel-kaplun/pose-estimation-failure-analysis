# Pose Estimation Failure Analysis

This repository focuses on a two-part independent research project studying the limits of human pose estimation models under occlusion.

---

## Research Question

How do modern pose estimation models fail under occlusion, viewpoint restriction, and domain shift?

---

## Experiment 1

Experiment 1 focuses on the fine-tuning of an HRNet-W18 model using annotated real-world videos of individuals performing barbell back squats, with an emphasis on occluded left shoulder keypoints. Despite careful annotation, augmentation, and configuration changes, the model failed to generalize, highlighting the limitations of pose estimation models when faced with viewpoint restriction, limited dataset size, and a narrowly scoped task.

---

## Experiment 2

Experiment 2 builds on these findings by creating a controlled synthetic dataset in Blender with systematically increasing occlusion and evaluating HRNet-W32 on this data. This experiment further highlights the limits of pose estimation models under controlled motion and occlusion.

---

## Overall Conclusion

Together, these two experiments illustrate practical limitations of modern pose estimation models in occlusion-heavy scenarios.

---

## Key Findings

- Fine-tuning HRNet-W18 on a narrow, single-viewpoint dataset did not improve generalization.  
- Synthetic control using Blender revealed that domain and pose shift degraded performance even without heavy occlusion.  
- At high occlusion levels, the model appeared to rely on learned body priors rather than image evidence.  

---

## My Role

Independent project: dataset creation, annotation, model configuration, training, and evaluation were all conducted by the author.

---

## Technical Work Involved

This project demonstrates:

- Dataset design and annotation workflows  
- Synthetic dataset generation in Blender  
- 3D-to-2D keypoint projection pipelines  
- Experimental control of occlusion variables  
- Failure-mode analysis of deep learning systems  

