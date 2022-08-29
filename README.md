# TIGER Algorithm

Algorithm for the [TIGER](https://tiger.grand-challenge.org/) challenge

Our algorithm ranked 1st in [Survival (Final) Evaluation](https://tiger.grand-challenge.org/evaluation/survival-final-evaluation/leaderboard/).

![leaderboard_screenshot](figure/leaderboard_screenshot.png)

## Tested Environment

`Ubuntu 20.04`

## Overview

### Overall pipeline

![pipeline](figure/pipeline.png)

### Method description

Detailed description of our method is found [here](figure/method_description.pdf).

## Summary of the modules

Used docker related codes and algorithm template from the [official example](https://github.com/DIAGNijmegen/pathology-tiger-algorithm-example).

#### `algorithm/`

entry point of the algorithm (modified official example code)


#### `configuration/`

contains codes for configuration

#### `detection/`

contains codes for detection model

#### `pipeline/`

contains codes for TILs inference pipeline

#### `pretrained weights/`

contains pretrained weights for detection and segmentation models

#### `segmentation/`

contains codes for segmentation model

#### `tils/`

contains codes to process for TILs inference

#### `./`

contains codes for docker
