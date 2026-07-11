# Drupella-YOLO: underwater dataset and code for object detection and quantification of *Drupella* spp.

Dataset, annotations, source code, model configurations, and post-detection quantification tools associated with the paper:

> **A deep learning-based approach for automated detection of small and occluded corallivorous *Drupella* spp. from coral reef imagery**

---

## Project

### Project background

Corallivorous *Drupella* snails can occur in dense aggregations on branching corals and may contribute to substantial loss of living coral tissue. Conventional underwater surveys rely heavily on visual inspection and manual counting, which are labor-intensive and may be affected by limited diving time, observer error, coral structural complexity, and partial target occlusion. This project combines underwater imagery, lightweight object detection, and post-detection quantification to support more efficient image-based monitoring of *Drupella* populations. The workflow was developed using data collected in collaboration with the New Heaven Reef Conservation Program in Koh Tao, Thailand.

### Main contributions

- A field dataset [drupella-dataset.zip](https://drive.google.com/file/d/1t53ldNqqKgMVdKANbA2-X6PaalZo1gXR/view?usp=sharing) containing both still photographs and video-extracted frames of *Drupella* snails in branching coral habitats.
- Bounding-box annotations for small, dense, and partially occluded underwater targets.
- Drupella-YOLO, a lightweight YOLOv13n-based detector incorporating:
  - Complementary Feature Complementary Mapping (CFCM);
  - Hierarchical Dual-Stream Attention (HDSA); and
  - Inverse Sample Module (ISM).
- Model-comparison and ablation experiments.
- A post-detection pipeline for:
  - image-level abundance estimation;
  - ruler-based pixel-to-millimeter conversion;
  - approximate body-length estimation; and
  - size-class classification.

---

## Overall workflow
The workflow consists of four stages:

1. Field survey and underwater image acquisition;
2. Dataset preparation and expert annotation;
3. Object detection using Drupella-YOLO; and
4. Post-detection abundance and scale-referenced size estimation.

<p align="center">
  <img src="images/flowchart.jpg" alt="Overall workflow of the proposed framework for automated *Drupella* spp. detection and post-detection quantification" width="95%">
</p>
---

## Dataset

### Study area and image acquisition

Underwater imagery was collected at two reef sites in Koh Tao, Thailand:

- Chalok Baan Kao Bay;
- Taa Chaa Bay.

Field surveys were conducted in:

- April 2023;
- August 2023;
- January 2024; and
- September 2024.

The mean sampling depth was approximately 5 m.

### Dataset composition

| Image source | Camera | Number of images | Original resolution |
|---|---|---:|---:|
| Still photographs | Olympus TG-6 | 96 | 4000 × 3000 pixels |
| Video-extracted frames | GoPro HERO12 Black | 204 | 3940 × 2160 pixels |
| **Total** | — | **300** | — |

Video frames were extracted at 2 s intervals. Visually similar frames were manually removed to reduce redundancy.

### Annotation protocol

All images were annotated using LabelImg v1.8.6.

- Annotation target: individual *Drupella* snails;
- Annotation type: bounding boxes.

### Dataset split

The dataset was split into training, validation, and test subsets containing 210, 30, and 60 images, respectively, following a ratio of 7:1:2.

### Dataset characteristics

The dataset represents several major challenges in underwater small-object detection:

- small target size;
- dense and overlapping aggregations;
- partial occlusion by branching corals;
- variable illumination and underwater visibility;
- complex coral reef backgrounds;
- visual similarity between snail shells, algae, coral skeletons, rubble, and other reef substrates.

### Dataset organization

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── metadata.txt
└── classes.txt
```

## Methodology

### Drupella-YOLO

YOLOv13n was selected as the baseline detector because of its lightweight architecture and real-time inference capability. Drupella-YOLO introduces three task-specific modifications:

#### CFCM

The Complementary Feature Complementary Mapping module is embedded in the backbone to strengthen the interaction between fine-grained spatial information and high-level semantic features.

#### HDSA

The Hierarchical Dual-Stream Attention module is inserted into the neck to enhance global channel dependencies and local spatial saliency before prediction.

#### ISM

The Inverse Sample Module replaces an original upsampling operation in the neck to improve the recovery of fine-grained target boundaries and shell-texture information during multi-scale feature fusion.

<p align="center">
  <img src="images/Drupella-YOLO-architecture.jpg" alt="Architecture of Drupella-YOLO" width="95%">
</p>

### Post-detection quantification

The post-detection pipeline converts model outputs into ecologically interpretable information.

For each image:

1. The number of detected bounding boxes is used as the image-level abundance.
2. Ruler tick points are manually annotated when a visible ruler is available.
3. A local pixel-to-millimeter conversion factor is calculated from the selected ruler segment.
4. The diagonal length of each detected bounding box is converted into an approximate physical body length.
5. Detected individuals are assigned to one of two size classes:
   - smaller than 2 cm;
   - 2 cm or larger.

These measurements should be interpreted as two-dimensional, scale-referenced estimates rather than direct three-dimensional shell measurements. Image-level abundance should not be interpreted as standardized population density because the original imagery was not collected using a fixed-area quadrat design.

---

## License
- **Source code:** [MIT License](LICENSE)
- **Dataset and annotations:** CC BY 4.0

---
# Citation
If this dataset or codes contributes to your research, please consider citing our paper:
```LaTeX
@article{shao2026drupellayolo,
title = {A deep learning-based approach for automated detection of small and occluded corallivorous *Drupella* spp. from coral reef imagery},
journal = {},
volume = {},
pages = {},
year = {2026},
issn = {},
doi = {},
url = {},
author = {Xinlei Shao and Jiaqi Wang and Kirsty Magson and Weitao Xu and Jundong Chen and Jun Sasaki and Fan Zhao}
}
```
# Q & A
For any questions, please [contact us.](mailto:yuishaoxinlei@gmail.com)
