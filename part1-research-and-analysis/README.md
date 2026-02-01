# Experiment 1: Research and Analysis

## Overview
This experiment investigates whether fine tuning an HRNet-18 model with an annotated, one-viewpoint dataset can significantly improve pose estimation under occlusion. 
## Motivation
This work was motivated by a personal interest of creating an app that helps people with their squat form. The problem I ran into was pose estimation failing on occluded limbs, making rule-based analysis impossible. 
## Dataset and Annotation
The dataset consisted of about 4000 frames. The left shoulder, left elbow, left hip, and left knee were annotated with annotator confidence for a narrowed scope. Different augmentations were also implemented. 
## Model and Training
The HRNet-18 architecture was tested because of its relatively low computational requirements. The model was initialized with ImageNet-pretrained weights and modified to output heatmaps for a reduced set of four keypoints.
## Results
Quantitative evaluation showed a low average precision and recall across all datasets. Across all runs, training loss dropped early, indicating the model stopped learning immediately.
## Limitations
The primary limitation was the size and scope of the dataset. 
## Artifacts
- `research-paper.pdf`: Full written report including methods, quantitative results, and discussion
- `research-poster.pdf`: Visual summary of the project and findings
