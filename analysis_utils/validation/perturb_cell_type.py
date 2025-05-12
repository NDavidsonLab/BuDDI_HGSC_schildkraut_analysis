from typing import Iterable, Optional, Tuple, List, Any

from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from ..buddi4data import BuDDI4Data
from .perturb_y import _perturb_y

def perturb_single_cell_type(
    obj: Any,
    data: BuDDI4Data,
    cell_type_col: str = 'cell_type',
    idx: Optional[Iterable[int]] = None,
    n_subsamples: Optional[int] = 500,
    n_resamples: int = 1,
    integrate_over: Optional[List[str]] = None,
    ignore_cell_types: Iterable[str] = [],
    seed: Optional[int] = 42,
):
    
    """
    Perform buddi4 model validation with cell type proportion perturbation experiment, 
    by swapping proportion matrices of pseudobulk entries corresponding to different cell types.

    :param obj: BuDDI4 model object.
    :param data: BuDDI4Data object containing the data.
    :param cell_type_col: Column name in metadata for cell types.
    :param idx: Indices of input data to use for perturbation. If None, all samples are used.
    :param n_subsamples: Number of subsamples to draw from each source cell type. 
        Should be relatively small (500-1000) to avoid memory issues.
    :param n_resamples: Number of resamples to draw from each source cell type.
        Should be 1 as re-sampling in single cell pseudo-bulk is largely unnecessary.
    :param integrate_over: List of latent spaces to integrate over.
        If None, no integration is performed.
    :param ignore_cell_types: List of cell types to ignore during perturbation.
    :param seed: Random seed for reproducibility.
    :return: Perturbed cell type proportions.
    """
    
    try:
        X=data.get(condition='kp', key='X')
        y=data.get(condition='kp', key='Y')
        meta=data.get(condition='kp', key='meta')
    except KeyError:
        raise ValueError("Data object must contain 'X', 'Y', and 'meta' keys under 'kp' condition.")
    
    try:
        cell_type_names = data.cell_type_names
    except AttributeError:
        raise ValueError("Data object must have 'cell_type_names' attribute.")

    if not cell_type_col in meta.columns:
        raise ValueError(f"Column '{cell_type_col}' not found in meta data.")
    if not all(ct in cell_type_names for ct in ignore_cell_types):
        raise ValueError(f"Some cell types in 'ignore_cell_types' not found in 'cell_type_names'.")

    return _perturb_y(
        obj=obj,
        X=X,
        y=y,
        meta=meta,
        cell_type_names=cell_type_names,
        cell_type_col=cell_type_col,
        idx=idx,
        n_subsamples=n_subsamples,
        n_resamples=n_resamples,
        integrate_over=integrate_over,
        ignore_cell_types=ignore_cell_types,
        seed=seed,
    )