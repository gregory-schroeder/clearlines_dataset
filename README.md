# ClearLines: A Dataset for Intrinsic Camera Calibration Using Straight Lines

![Dataset Example](dataset/images/title_image.JPG)

## Overview
ClearLines is a dataset designed for intrinsic camera calibration tasks in real-world outdoor scenarios. The dataset focuses on edge-segment detection and includes labeled straight edge-segments. It provides a foundation for evaluating and advancing calibration algorithms, particularly for distortion parameter estimation.

This repository contains:
- The **ClearLines dataset**, sourced from KITTI and IAMCV.
- Evaluation tools and metrics, including precision and recall calculations for edge-segment detection.

---

## Getting Started

### Installation
Clone this repository and install dependencies (with Git LFS enabled):
```bash
git lfs install
git clone https://github.com/gregory-schroeder/clearlines_dataset.git
cd clearlines_dataset
pip install -r requirements.txt
```


## Dataset Structure

The dataset is organized as follows:

```
dataset/
├── images/               # Original images
├── labels/               # Manual and pipeline-generated labels
```

- **images/**: Original input images collected from KITTI and IAMCV.
- **labels/**: Manual annotated straight edge-segments.

---

## Evaluation

### Evaluate Results
Evaluate the detected edge-segments against ground-truth labels:
```bash
python evaluate.py --ground-truth dataset/labels --predictions my_results/
```

### Metrics Provided:
- **Precision**: Proportion of detected edge-segments that are true positives.
- **Recall**: Proportion of true edge-segments successfully detected.
- **F1-Score**: The harmonic mean of precision and recall.
- **Fine-grained Metrics**:
  - Intersection over Union (IoU)
  - Average Rotation Error
  - Average Translation Error


---

### Evaluation Output

The evaluation script generates results in the `evaluation/` directory:

```
evaluation/
├── summary.txt            # Overall metrics: precision, recall, F1-Score
├── individual_results/    # Per-image IoU, rotation, and translation errors
└── visualization/         # Ground-truth vs detected edge-segment visuals
```

- **summary.txt**: Dataset-wide metrics.
- **individual_results/**: Detailed metrics for each image.
- **visualization/**: Visual comparisons of ground-truth and predictions. Green: True positives, Red: False positives, Orange: False negatives.

These outputs help analyze algorithm performance and refine pipeline parameters.


## Citation

If you use the ClearLines dataset in your research, please cite:

```bibtex
@article{schroeder2025clearlines,
  title={ClearLines - Camera Calibration from Straight Lines},
  author={Gregory Schroeder and Mohamed Sabry and Cristina Olaverri-Monreal},
  journal={2025 IEEE Intelligent Vehicles Symposium (IV)},
  year={2025}
}
```





