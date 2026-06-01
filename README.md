# Active Fire Detection using MTG FCI

## Overview
This repository provides implementations of machine learning (ML) and deep learning (DL) models to predict active fires using Meteosat Third Generation (MTG) Flexible Combined Imager (FCI) data. Users can choose their preferred modeling approach, load the pre-trained models, and implement them for predictive tasks.

## Data 
The `data/` directory contains sample MTG FCI imagery covering Cyprus, Israel, and Syria. All data has been preprocessed using the `satpy` package. 

The data folder includes the following files:
* **Multispectral Image:** A 4-band composite at 1km spatial resolution, stacked in the exact following order: VIS 0.6, NIR 2.2, IR 3.8, and IR 10.5.
* **Cloud Mask (CLM):** The official MTG CLM product at 1km resolution. 

The CLM product is required as an input feature for the ML model but is not used for the DL models.

## Models
The repository includes the following architectures for fire prediction:

### Machine Learning
* **CatBoost:** A gradient boosting model utilizing both the 4-band multispectral image and the CLM product as inputs.

### Deep Learning
The DL models rely solely on the 4-band multispectral image. 
* **MAnet:** Configured with a Mix Vision Transformer encoder.
* **Unet++:** Configured with an `se_resnext50_32x4d` encoder.

## Installation

You can replicate the environment using either Conda (Option 1) or Pip (Option 2).

### Option 1: Conda Environment File (Recommended)
This method automatically configures Python 3.9.23 and installs all dependencies using the provided configuration file.

```bash
# Create the environment from the environment.yml file
conda env create -f environment.yml

# Activate the environment
conda activate mtg_fire_env
```

### Option 2: Pip Requirements File
This method requires you to create the Python environment manually before installing the libraries via the requirements list.

```bash
# Create a new conda environment with the specific Python version
conda create --name mtg_fire_env python=3.9.23 -y

# Activate the environment
conda activate mtg_fire_env

# Install the required packages via pip
pip install -r requirements.txt
```