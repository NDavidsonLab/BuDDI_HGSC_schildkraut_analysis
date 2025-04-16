from typing import Iterable, Optional, List

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model

def perturb_cell_type(
    decoder: Model,
    z: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    cell_type_col: str = 'cell_type',
    all_cell_types: Optional[List[str]]=None,
    idx: Optional[Iterable] = None,
    n_subsamples: int = 500
):
    
    """
    """

    if all_cell_types is None:
        all_cell_types = list(meta[cell_type_col].unique())
    
    if idx is None:
        idx = np.arange(z.shape[0])
    
    # Subset the data to the specifiied indices/condition/sample id
    _z = z[idx, :]
    _y = y[idx, :]
    _meta = meta.iloc[idx, :].copy()

    # Create mapping of cell type to indices
    cell_type_to_idx = {
    cell_type: np.where(_meta['cell_type'] == cell_type)[0] \
            for cell_type in all_cell_types
        }
    # Create mapping of cell type to corresponding y (proportions)
    cell_type_to_prop = {
        cell_type: _y[idx,:] \
            for cell_type, idx in cell_type_to_idx.items()
        }
    
    xs_reconst_ct_perturb = []
    ct_perturb_metas = []
    for _ct_source in all_cell_types:
        for _ct_target in all_cell_types:
            if _ct_source == _ct_target:
                continue

            source_idx = cell_type_to_idx[_ct_source]
            source_idx = source_idx[np.random.choice(
                np.arange(len(source_idx)), n_subsamples, replace=True)]

            source_z = z[source_idx,:]
            # subsample y to ensure compatibility and shortern computation time
            target_y = cell_type_to_prop[_ct_target]
            target_y = target_y[np.random.choice(
                np.arange(len(target_y)), n_subsamples, replace=True),:]

            # compute the decoding after perturbation
            with tf.device('/CPU:0'):
                xs_reconst_ct_perturb.append(
                    decoder(
                        np.concatenate(
                            [target_y, source_z],
                            axis=1
                        )
                    ).numpy()
                )
            # create the metadata for the perturbed cell type
            ct_perturb_metas.append(
                pd.DataFrame(
                    data = {
                        'cell_type': [_ct_source] * n_subsamples,
                        'cell_type_perturb_target': [_ct_target] * n_subsamples,
                    }
                )
            )

    x_reconst_ct_perturb = np.concatenate(xs_reconst_ct_perturb, axis=0)
    ct_perturb_meta = pd.concat(ct_perturb_metas, axis=0)

    return x_reconst_ct_perturb, ct_perturb_meta