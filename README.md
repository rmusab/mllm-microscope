# MLLM-Microscope

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-B31B1B)](https://arxiv.org/abs/1234.56789)

## Overview

This repository contains the implementation of the `MllmMicroscope` class, designed to analyze and visualize intermediate embeddings from multimodal large language models (MLLMs). Currently, the code supports models like LLaVA-NeXT and OmniFusion, and includes methods for extracting embeddings, calculating various metrics, and visualizing the results.

## Features

- Supports LLaVA-NeXT and OmniFusion models.
- Extracts all **intermediate embeddings** from the text and image tokens.
- Calculates **Procrustes similarity** between layers to measure the linearity of transformations.
- Computes the **intrinsic dimension** and **anisotropy** of embeddings.
- Provides **visualization** for intermediate embeddings using SVD and t-SNE.

## Demo Jupyter Notebook

The repository includes a full demo Jupyter Notebook applying both models to the ScienceQA dataset: `mllm_microscope_demo.ipynb`.

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/rmusab/mllm-microscope.git
    cd llm-microscope
    ```

2. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Initialization

```python
from llm_microscope import MllmMicroscope

# Initialize the microscope for a specific device
device = "cuda:0"
microscope = MllmMicroscope(device=device)
```

### Extracting Embeddings

```python
# List of texts and corresponding images (if any)
texts = ["This is a sample text.", "Another text input."]
images = [None, None]  # Replace None with actual image data if available

# Extract embeddings for a specific model
model_name = "LLaVA-NeXT"
embeddings = microscope.get_intermediate_embeddings(model_name, texts, images)
```

### Analyzing Embeddings

# Analyze the extracted embeddings
```python
microscope.analyze_embeddings(model_name)
```

### Plotting Results

```python
# Plot the analysis results
microscope.plot_results(save=True, saving_path="./results")
```

### Visualizing Embeddings

```python
# Visualize intermediate embeddings
microscope.visualize_intermediate_embeddings(save=True, saving_path="./visualizations")
```

## Citing Our Work

If you find our research useful in your work, please cite our work:

```bibtex
@misc{Mussabayev2024-mllmmicro,
  author = {Ravil Mussabayev and Andrey Kuznetsov},
  title = {MLLM-Microscope: Unlocking Hidden Structure Within Multimodal Large Language Models},
  year = {2024},
  archivePrefix = {arXiv},
  eprint = {1234.56789},
  primaryClass = {cs.CL}
}
```