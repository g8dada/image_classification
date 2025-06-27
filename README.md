# About Dataset

## Context

This is image data of Natural Scenes around the world.

## Content

This Data contains around 25k images of size 150x150 distributed under 6 categories.

```{'buildings' -> 0,
'forest' -> 1,
'glacier' -> 2,
'mountain' -> 3,
'sea' -> 4,
'street' -> 5 }
```

The Train, Test and Prediction data is separated in each zip files. There are around 14k images in Train, 3k in Test and 7k in Prediction.

This data was initially published on https://datahack.analyticsvidhya.com by Intel to host a Image classification Challenge.

## Acknowledgements

Thanks to https://datahack.analyticsvidhya.com for the challenge and Intel for the Data

Photo by Jan Böttinger on Unsplash

## Inspiration

Want to build powerful Neural network that can classify these images with more accuracy.






# PyTorch Image Classification

A complete PyTorch implementation for image classification using transfer learning with ResNet architecture.

## Features

- **Transfer Learning**: Uses pre-trained ResNet models
- **Data Augmentation**: Comprehensive image augmentations for better generalization
- **Early Stopping**: Prevents overfitting with patience-based early stopping
- **Tensorboard Logging**: Real-time training monitoring
- **Checkpointing**: Saves best and final models
- **Visualization**: Plots training history

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt