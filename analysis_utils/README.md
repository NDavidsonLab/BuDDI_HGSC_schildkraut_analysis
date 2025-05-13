# BuDDI4 Analysis Utilities

This repository contains Python utility modules for analyzing, perturbing, reconstructing, and visualizing data using the BuDDI4 framework — a model for deconvoluting bulk RNA-seq data into cell-type-specific expression profiles.

## Module File Structure

```
analysis_utils/
├── buddi4data.py          # Data container class for BuDDI4 datasets
├── __init__.py            # Package initializer
├── README.md              # Module documentation (this file)
└── validation/
    ├── __init__.py        # Subpackage initializer
    ├── perturb_cell_type.py # Cell-type perturbation functions
    ├── plot_perturb.py    # Visualization tools (PCA/UMAP, heatmaps)
    └── resample.py        # Reconstruction and resampling functions
```

---

## Module Overview

* **`buddi4data.py`** → Data container class
* **`validation/perturb_cell_type.py`** → Cell-type perturbation functions
* **`validation/resample.py`** → Reconstruction and resampling functions
* **`validation/plot_perturb.py`** → Visualization tools (dimensionality reduction plots, correlation heatmaps)

---

## Installation

Not installable for now.

---

## Module Details

### `buddi4data.py`

Defines the `BuDDI4Data` class:

* Organizes input features (`X`), targets (`Y`), labels, and metadata
* Separates known (`kp`) vs. unknown (`unkp`) proportion datasets
* Provides methods for length, representation, and data access

### `validation/perturb_cell_type.py`

Main function: `perturb_cell_type()`

* Perturbs dominant cell types in the dataset
* Generates reconstructed outputs for source → target cell-type swaps

### `validation/resample.py`

Main function: `reconstruct()`

* Performs stochastic reconstruction from latent space
* Optionally repeats sampling (`n_resamples`)
* Optionally replaces `Y` input with random uniform softmax-normalized values

### `validation/plot_perturb.py`

Main functions:

* `plot_perturb_reduction()` → PCA or UMAP scatterplots comparing baseline and perturbed data
* `plot_corr_matrix_with_metadata()` → Sample-sample correlation heatmap comparing ground truth and reconstructed data, sorted by metadata category

---

## Example Usage
See notebooks under `3.validate_trained_buddi4`