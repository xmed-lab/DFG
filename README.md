# Leveraging Segment Anything Model for Source-Free Domain Adaptation via Dual Feature Guided Auto-Prompting
This repository contains Pytorch implementation of our source-free domain adaptation (SFDA) method with Dual Feature Guided (DFG) auto-prompting approach.

<!-- ![method](./figures/method.png "")
## Introduction
[Context-Aware Pseudo-Label Refinement for Source-Free Domain Adaptive Fundus Image Segmentation](https://arxiv.org/pdf/2308.07731.pdf), MICCAI 2023

In the domain adaptation problem, source data may be unavailable to the target client side due to privacy or intellectual property issues. Source-free unsupervised domain adaptation (SF-UDA) aims at adapting a model trained on the source side to align the target distribution with only the source model and unlabeled target data. The source model usually produces noisy and context-inconsistent pseudo-labels on the target domain, i.e., neighbouring regions that have a similar visual appearance are annotated with different pseudo-labels. 
This observation motivates us to refine pseudo-labels with context relations. Another observation is that features of the same class tend to form a cluster despite the domain gap, which implies context relations can be readily calculated from feature distances. To this end, we propose a context-aware pseudo-label refinement method for SF-UDA. Specifically, a context-similarity learning module is developed to learn context relations. Next, pseudo-label revision is designed utilizing the learned context relations. Further, we propose calibrating the revised pseudo-labels to compensate for wrong revision caused by inaccurate context relations. Additionally, we adopt a pixel-level and class-level denoising scheme to select reliable pseudo-labels for domain adaptation. Experiments on cross-domain fundus images indicate that our approach yields the state-of-the-art results. -->

## Installation
Create the environment from the `environment.yml` file:
```
conda env create -f environment.yml
conda activate dfg
```
## Data preparation
* Download the BTCV dataset from [MICCAI 2015 Multi-Atlas Abdomen Labeling Challenge](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789), and the CHAOS dataset from [2019 CHAOS Challenge](https://chaos.grand-challenge.org/). Then preprocess the downloaded data referring to `./preprocess.ipynb`.
You can also directly download our preprocessed datasets from [here](https://drive.google.com/drive/folders/1g2ar0L18ryO9zlmVnl-1Ia-XrHkDODfN?usp=sharing).

## Training
The following are the steps for the CHAOS (MRI) to BTCV (CT) adaptation.
* Download the source domain model from [here](https://drive.google.com/file/d/18zhjTuy3LFqWMPckrhby8SX9-2muh1KQ/view?usp=sharing) or specify the data path in `configs/train_source_seg.yaml` and then run 
```
python main_trainer_source.py --config_file configs/train_source_seg.yaml --gpu_id 0
```
* Download the trained model after the feature aggregation phase from [here](https://drive.google.com/file/d/1eOOnQ4Je9UrfJ-Hqf5e9aaHyUTK6-iMv/view?usp=sharing) or specify the source model path and data path in `configs/train_target_adapt_FA.yaml`, and then run
```
python main_trainer_fa.py --config_file configs/train_target_adapt_FA.yaml --gpu_id 0
```
* Download the MedSAM model checkpoint from [here](https://drive.google.com/file/d/1khQO5G-qYZsCkocEhZ-HhX8IioEUAZ9Q/view?usp=sharing) and put it under `./medsam/work_dir/MedSAM`.
* Specify the model (after feature aggregation) path, data path, and refined pseudo-label paths in `configs/train_target_adapt_SAM.yaml`, and then run
```
python main_trainer_sam.py --config_file configs/train_target_adapt_SAM.yaml --gpu_id 0
```
<!--
## Result
cup: 0.7503 disc: 0.9503 avg: 0.8503 cup: 9.8381 disc: 4.3139 avg: 7.0760
## REFUGE to Drishti-GS adaptation
Follow the same pipeline as above, but run these commands to specify the new parameters:
```
python train_source.py --datasetS Domain4
python generate_pseudo.py --dataset Domain1 --model-file /path/to/source_model
python sim_learn.py --dataset Domain1 --model-file /path/to/source_model --pseudo /path/to/pseudo_label
python pl_refine.py --dataset Domain1 --weights /path/to/context_similarity_model --logt 5 --pseudo /path/to/pseudo_label
python train_target.py --dataset Domain1 --model-file /path/to/context_similarity_model --num_epochs 20
```

## Acknowledgement
We would like to thank the great work of the following open-source projects: [DPL](https://github.com/cchen-cc/SFDA-DPL), [AffinityNet](https://github.com/jiwoon-ahn/psa). -->