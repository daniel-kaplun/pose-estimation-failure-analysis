# Experiment 1: Research and Analysis

## Overview
This experiment investigates whether fine-tuning an HRNet-18 model with an annotated, single-viewpoint dataset can significantly improve pose estimation under occlusion.

## Motivation
This work was motivated by a personal interest in creating an app that helps people improve their squat form. The primary challenge encountered was pose estimation failing on occluded limbs, making rule-based analysis unreliable.

## Dataset and Annotation
The dataset consisted of approximately 4,000 frames extracted from real-world squat videos. The left shoulder, left elbow, left hip, and left knee were annotated with annotator confidence scores to support a narrowed scope. Several data augmentations were also applied.

## Model and Training
The HRNet-18 architecture was selected due to its relatively low computational requirements. The model was initialized with ImageNet-pretrained weights and modified to output heatmaps for a reduced set of four keypoints.

## Results
Quantitative evaluation showed low average precision and recall across all datasets. Across all runs, training loss dropped rapidly within the first epochs, indicating that the model failed to learn meaningful representations beyond early training.

## Limitations
The primary limitation was the size and scope of the dataset. Restricting the viewpoint and reducing the number of keypoints limited the model’s ability to generalize, even with augmentation.

## Artifacts
- `research-paper.pdf`: Full written report including methods, quantitative results, and discussion
- `research-poster.pdf`: Visual summary of the project and findings
