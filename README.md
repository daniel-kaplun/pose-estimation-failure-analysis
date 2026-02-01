This repository focuses on a two-part independent research project studying the limits of human pose estimation models under occlusion.

Experiment 1 focuses on the fine-tuning of an HRNet-18 model using annotated real-world videos of individuals performing barbell back squats, with an emphasis on occluded left shoulder keypoints. Despite careful annotation, augmentation, and configuration changes, the model failed to generalize, highlighting the limitations of pose estimation models when faced with viewpoint restriction, limited dataset size, and a narrowly scoped task.

Experiment 2 builds on these findings by creating a controlled synthetic dataset in Blender with systematic occlusion and evaluating HRNet-32 on this data. This experiment further highlights the limits of pose estimation models under controlled motion and occlusion.

Together, these two experiments illustrate practical limitations of modern pose estimation models in occlusion-heavy scenarios.
