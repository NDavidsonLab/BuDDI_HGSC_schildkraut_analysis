from typing import Iterable, Optional, Tuple, List, Any
import warnings

from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from .perturb_z import _perturb_z

def perturb_bulk_sample(
    obj: Any,
    data: Any,
    sample_col: str = 'sample_id',
    sample_branch_name: str = 'label',
    idx: Optional[Iterable[int]] = None,
    n_subsamples: Optional[int] = None,
    n_resamples: int = 100,
    ignore_samples: Iterable[str] = [],
    integrate_over_cell_types: bool = True,
    integrate_over: Optional[List[str]] = None,
    seed: Optional[int] = 42,
):
    """
    Perform buddi4 model validation by perturbing the sample latent space of the model.

    :param obj: BuDDI4 model object.
    :param data: BuDDI4Data object containing the data.
    :param sample_col: Column name in metadata for the sample ID.
    :param sample_branch_name: Branch name in the encoder for the sample latent space.
    :param idx: Indices of input data to use for perturbation. 
        If None, all samples are used. Highly recommend to supply an index list of only few samples
        to avoid memory issues and long computation time.
    :param n_subsamples: Number of subsamples to draw from each source sample.
        If None, uses all entries. Recommended to use None or 1 for bulk data since each sample 
        should only correspond to one entry.
    :param n_resamples: Number of resamples to draw from each perturbation. 
        Recommended to use 100 or larger for bulk data.
    :param ignore_samples: List of sample IDs to ignore during perturbation.
    :param integrate_over_cell_types: Whether to integrate over cell types.
    :param integrate_over: List of latent spaces to integrate over.
        If None, no integration is performed. Should not include the branch to be perturbed.
    :param seed: Random seed for reproducibility.
    :return: Tuple of reconstructed data and metadata.
        - x_reconst_sample_perturb: Reconstructed data after perturbation.
        - sample_perturb_meta: Metadata for the perturbation.
    """
    
    try:
        X=data.get(condition='unkp', key='X')
        y=data.get(condition='unkp', key='Y')
        meta=data.get(condition='unkp', key='meta')
    except KeyError:
        raise ValueError("Data object must contain 'X', 'Y', and 'meta' keys.")
    
    if sample_col not in meta.columns:
        raise ValueError(f"Column '{sample_col}' not found in meta data.")
    
    if sample_branch_name not in obj.encoder_branch_names:
        raise ValueError(f"Branch '{sample_branch_name}' not found in encoder branches.")
    
    if integrate_over is not None:
        if not all(branch_name in obj.encoder_branch_names for branch_name in integrate_over):
            raise ValueError(f"Branches {integrate_over} not found in encoder branches.")
        
        if sample_branch_name in integrate_over:
            raise ValueError(f"Branch {sample_branch_name} to be perturbed cannot be integrated over.")

    return _perturb_z(
        obj=obj,
        X=X,
        y=y,
        meta=meta,
        z_col_name=sample_col,
        z_branch_name=sample_branch_name,
        idx=idx,
        n_subsamples=n_subsamples,
        n_resamples=n_resamples,
        integrate_over_y=integrate_over_cell_types,
        integrate_over=integrate_over,
        ignore_class=ignore_samples,
        seed=seed,
    )

def perturb_single_cell_sample(
    obj: Any,
    data: Any,
    sample_col: str = 'sample_id',
    sample_branch_name: str = 'label',
    n_subsamples: Optional[int] = 500,
    n_resamples: int = 100,
    ignore_samples: Iterable[str] = [],
    integrate_over_cell_types: bool = False,
    integrate_over: Optional[List[str]] = None,
    seed: Optional[int] = 42,
    query_kwargs: Optional[dict] = None,
):
    """
    Perform buddi4 model validation by perturbing the sample latent space of the model.

    :param obj: BuDDI4 model object.
    :param data: BuDDI4Data object containing the data.
    :param sample_col: Column name in metadata for the sample ID.
    :param sample_branch_name: Branch name in the encoder for the sample latent space.
    :param idx: Indices of input data to use for perturbation. 
        If None, all samples are used. Highly recommend to supply an index list of only few samples
        to avoid memory issues and long computation time.
    :param n_subsamples: Number of subsamples to draw from each source sample.
        If None, uses all entries. Recommended to use 500-1000 for single cell data.
    :param n_resamples: Number of resamples to draw from each perturbation.
        Recommended to use 1 for single cell data as each sample corresponds to large number of entries.
    :param ignore_samples: List of sample IDs to ignore during perturbation.
    :param integrate_over_cell_types: Whether to integrate over cell types.
    :param integrate_over: List of latent spaces to integrate over.
        If None, no integration is performed. Should not include the branch to be perturbed.
    :param seed: Random seed for reproducibility.
    :return: Tuple of reconstructed data and metadata.
        - x_reconst_sample_perturb: Reconstructed data after perturbation.
        - sample_perturb_meta: Metadata for the perturbation.
    """
    
    query_kwargs = query_kwargs or {}
    if 'samp_type' in query_kwargs:
        if query_kwargs['samp_type'] != 'sc_ref':
            warnings.warn("Overriding query_kwargs['samp_type'] to 'sc_ref'")
            query_kwargs['samp_type'] = 'sc_ref'
    else:
        query_kwargs['samp_type'] = 'sc_ref'

    try:
        data_query = data.query(
            **query_kwargs
        )
    except Exception as e:
        raise ValueError(f"Error querying data: {e}")
    
    try:
        (X, y, meta) = data_query.get(
            ('X', 'Y', 'meta')
        )
    except Exception as e:
        raise ValueError(f"Error retrieving data from the BuDDI4Data object: {e}")
    
    if sample_col not in meta.columns:
        raise ValueError(f"Column '{sample_col}' not found in meta data.")
    
    if sample_branch_name not in obj.encoder_branch_names:
        raise ValueError(f"Branch '{sample_branch_name}' not found in encoder branches.")
    
    if integrate_over is not None:
        if not all(branch_name in obj.encoder_branch_names for branch_name in integrate_over):
            raise ValueError(f"Branches {integrate_over} not found in encoder branches.")
        
        if sample_branch_name in integrate_over:
            raise ValueError(f"Branch {sample_branch_name} to be perturbed cannot be integrated over.")

    return _perturb_z(
        obj=obj,
        X=X,
        y=y,
        meta=meta,
        z_col_name=sample_col,
        z_branch_name=sample_branch_name,
        n_subsamples=n_subsamples,
        n_resamples=n_resamples,
        integrate_over_y=integrate_over_cell_types,
        integrate_over=integrate_over,
        ignore_class=ignore_samples,
        seed=seed,
    )