# Leveraging Segment Anything Model for Source-Free Domain Adaptation via Dual Feature Guided Auto-Prompting
This repository contains Pytorch implementation of our source-free domain adaptation (SFDA) method with Dual Feature Guided (DFG) auto-prompting approach.

<!-- ![method](./figures/method.png "") -->
## Introduction
Source-free domain adaptation (SFDA) for segmentation aims at adapting a model trained in the source domain to perform well in the target domain with only the source model and unlabeled target data. Inspired by the recent success of Segment Anything Model (SAM) which exhibits the generality of segmenting images of various modalities and in different domains given human-annotated prompts like bounding boxes or points, we for the first time explore the potentials of Segment Anything Model for SFDA via automatedly finding an accurate bounding box prompt. We find that the bounding boxes directly generated with existing SFDA approaches are defective due to the domain gap. To tackle this issue, we propose a novel Dual Feature Guided (DFG) auto-prompting approach to search for the box prompt. Specifically, the source model is first trained in a feature aggregation phase, which not only preliminarily adapts the source model to the target domain but also builds a feature distribution well-prepared for box prompt search. In the second phase, based on two feature distribution observations, we gradually expand the box prompt with the guidance of the target model feature and the SAM feature to handle the class-wise clustered target features and the class-wise dispersed target features, respectively. To remove the potentially enlarged false positive regions caused by the over-confident prediction of the target model, the refined pseudo-labels produced by SAM are further postprocessed based on connectivity analysis. 
Experiments on 3D and 2D datasets indicate that our approach yields superior performance compared to conventional methods.

## Installation
Create the environment from the `environment.yml` file:
```
conda env create -f environment.yml
conda activate dfg
```
## Data preparation
* Download the BTCV dataset from [MICCAI 2015 Multi-Atlas Abdomen Labeling Challenge](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789), and the CHAOS dataset from [2019 CHAOS Challenge](https://chaos.grand-challenge.org/). Then preprocess the downloaded data referring to `./preprocess.ipynb`.
You can also directly download our preprocessed datasets from [here](https://drive.google.com/drive/folders/1g2ar0L18ryO9zlmVnl-1Ia-XrHkDODfN?usp=sharing). The paths to the datasets need to be specified in the yaml files in `./configs`.

## Training
The following are the steps for the CHAOS (MRI) to BTCV (CT) adaptation.
* Download the source domain model from [here](https://drive.google.com/file/d/18zhjTuy3LFqWMPckrhby8SX9-2muh1KQ/view?usp=sharing) or specify the data path in `configs/train_source_seg.yaml` and then run 
```
python main_trainer_source.py --config_file configs/train_source_seg.yaml
```
* Download the trained model after the feature aggregation phase from [here](https://drive.google.com/file/d/1eOOnQ4Je9UrfJ-Hqf5e9aaHyUTK6-iMv/view?usp=sharing) or specify the source model path and data path in `configs/train_target_adapt_FA.yaml`, and then run
```
python main_trainer_fa.py --config_file configs/train_target_adapt_FA.yaml
```
* Download the MedSAM model checkpoint from [here](https://drive.google.com/file/d/1khQO5G-qYZsCkocEhZ-HhX8IioEUAZ9Q/view?usp=sharing) and put it under `./medsam/work_dir/MedSAM`.
* Specify the model (after feature aggregation) path, data path, and refined pseudo-label paths in `configs/train_target_adapt_SAM.yaml`, and then run
```
python main_trainer_sam.py --config_file configs/train_target_adapt_SAM.yaml
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