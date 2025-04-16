from typing import Iterable, Optional, List

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model

def perturb_sample_id(
    decoder: Model,
    z: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    sample_id_col: str = 'sample_id',
    idx: Optional[Iterable] = None,
    n_subsamples: int = 500,
    integrate_y: bool = False,
):
    """
    """

    if idx is None:
        idx = np.arange(z.shape[0])
    
    # Subset the data to the specifiied indices/condition/sample id
    _z = z[idx, :]
    if integrate_y:
        _y = np.mean(y[idx, :], axis=0, keepdims=True)
        _y = np.tile(_y, (len(idx), 1))
    else:
        _y = y[idx, :]
    _meta = meta.iloc[idx, :].copy()

    # Create mapping of sample id to indices
    sample_id_to_idx = {
        sample_id: np.where(_meta['sample_id'] == sample_id)[0] \
                for sample_id in _meta['sample_id'].unique()
            }
    
    xs_reconst_sample_perturb = []
    sample_perturb_metas = []
    for _sample_source in _meta['sample_id'].unique():
        for _sample_target in _meta['sample_id'].unique():
            if _sample_source == _sample_target:
                continue

            source_idx = sample_id_to_idx[_sample_source]
            source_idx = source_idx[np.random.choice(
                np.arange(len(source_idx)), n_subsamples, replace=True)]

            source_z = z[source_idx,:]
            

