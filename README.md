# BEIRA: Brain EEG-fMRI Reconstruction Autoencoder

BEIRA is a deep learning project for reconstructing fMRI region-of-interest signals from EEG time-series data. The project uses an interpretable 1D convolutional autoencoder built in PyTorch to learn temporal and spatial EEG representations and predict subcortical BOLD activity from the CWL simultaneous EEG-fMRI dataset.

The repository contains the model architecture, preprocessing utilities, dataset wrappers, training helpers, inference utilities, and experiment notebooks used for final-year project work.

## Project Highlights

- EEG-to-fMRI sequence regression using PyTorch.
- Interpretable EEG front-end with learnable spatial filtering, band-pass filtering, envelope extraction, and low-pass filtering.
- 1D convolutional encoder-decoder architecture for temporal signal reconstruction.
- Multi-head autoencoder variant for predicting multiple fMRI ROIs.
- CWL dataset helpers for EEG loading, fMRI ROI extraction, interpolation, alignment, and preprocessing.
- Correlation-based evaluation and visualization utilities for predicted versus true ROI signals.

## Repository Structure

```text
BEIRA/
|-- autoencoder_new_Artur.py      # Autoencoder and multi-head model definitions
|-- get_datasets.py               # CWL EEG-fMRI loading and ROI extraction helpers
|-- preproc.py                    # EEG/fMRI preprocessing and alignment utilities
|-- torch_dataset.py              # Sliding-window PyTorch dataset wrapper
|-- train_utils.py                # Losses, training loop, checkpointing, W&B logging
|-- inference.py                  # Inference metrics and visualization functions
|-- main.ipynb                    # Training and experimentation notebook
|-- BEIRA_Inference.ipynb         # Inference and interpretation notebook
|-- env_torch.yml                 # Original Conda environment export
|-- requirements.txt              # Lightweight Python dependency list
`-- README.md
```

## Dataset Notice

The raw CWL EEG-fMRI dataset and neuroimaging files are not included in this GitHub repository because they are large binary research artifacts. Keep the dataset locally under:

```text
Dataset/
`-- trio1/
    `-- CWL_Data/
```

The code expects paths such as:

```text
Dataset/trio1/CWL_Data/eeg/in-scan/
Dataset/trio1/CWL_Data/mri/epi_normalized/
Dataset/trio1/CWL_Data/mri/epi_motionparams/
```

If you need to share trained checkpoints or raw data, use a release asset, cloud storage, or Git LFS instead of committing large files directly.

## Installation

Create a Python environment and install the required libraries:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Alternatively, recreate the original Conda environment:

```bash
conda env create -f env_torch.yml
conda activate myenv_torch
```

## Workflow

1. Place the CWL dataset in the local `Dataset/` directory.
2. Open `main.ipynb` for preprocessing, model configuration, training, and validation.
3. Use `BEIRA_Inference.ipynb` to load the trained model, run inference, inspect predictions, and visualize ROI correlations.

## Model Overview

The core architecture is implemented in `autoencoder_new_Artur.py`.

The model first applies an interpretable EEG processing block:

- `Conv1d` spatial unmixing across electrodes.
- Depthwise temporal band-pass filtering.
- Batch normalization and ReLU envelope extraction.
- Depthwise low-pass filtering.

The processed signal is then passed through a 1D convolutional encoder-decoder that downsamples and upsamples the sequence before producing ROI-level fMRI predictions. The multi-head version trains a separate output head per ROI.

## Evaluation

The inference utilities report:

- Pearson correlation per ROI.
- Average ROI correlation.
- Predicted versus true fMRI time-series plots.
- Grouped left/right ROI correlation bar plots.

The included experiment artifacts show a validation correlation checkpoint around `0.4325` for the saved run.

## Main Technologies
- Python
- PyTorch
- MNE
- Nilearn
- NumPy / Pandas / SciPy
- Matplotlib / Seaborn
- Weights & Biases
