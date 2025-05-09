from typing import Iterable, Optional, Tuple, List, Any

from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from ..buddi4data import BuDDI4Data

def perturb_sample(
    obj: Any,
    data: BuDDI4Data,
    sample_col: str = 'sample_id',
    sample_branch_name: str = 'label',
    idx: Optional[Iterable[int]] = None,
    n_subsamples: int = 500,
    seed: Optional[int] = 42,
    ignore_samples: Iterable[str] = [],
    integrate_over_cell_types: bool = True,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Perturb samples in your data and get reconstructions.

    :param obj: BuDDI4 model object.
    :param data: BuDDI4Data object.
    :param sample_col: Column name in meta data for samples.
    :param sample_branch_name: Branch name for the sample encoder.
    :param idx: Indices of the data to use for perturbation.
    :param n_subsamples: Number of subsamples to use for each perturbation.
    :param seed: Random seed for reproducibility.
    :param ignore_samples: Samples to ignore during perturbation.
    :param integrate_over_cell_types: Whether to integrate over cell types.
    :return: Tuple of reconstructed data and metadata.
        - x_reconst_sample_perturb: Reconstructed data after perturbation.
        - sample_perturb_meta: Metadata for the perturbation.
    """

    rng = np.random.default_rng(seed)
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)

    # 1) Gather & subset data if idx is provided
    X = data.get('kp', 'X')
    y = data.get('kp', 'Y')
    meta = data.get('kp', 'meta')
    if idx is None:
        idx = np.arange(X.shape[0])
    
    X_sub, y_sub = X[idx,:], y[idx,:]
    meta_sub = meta.iloc[idx].copy()
    # integrate over cell types if specified
    if integrate_over_cell_types:
        y_sub_mean = np.mean(y_sub, axis=0, keepdims=True)
        y_sub_mean_expanded = np.broadcast_to(y_sub_mean, y_sub.shape)
        y_sub = y_sub_mean_expanded

    # 2) Build concatenated z_sub
    branch_names = list(obj.encoder_branch_names)
    if 'slack' in obj.encoders:
        branch_names += ['slack']

    # 3) Map samples to indices & ys
    sample_names = meta_sub[sample_col].unique()
    sample_to_idx = {sample: np.where(meta_sub[sample_col] == sample)[0]
                     for sample in sample_names}
    valid_samples = [sample for sample, rows in sample_to_idx.items() if rows.size > 0]
    valid_samples = [sample for sample in valid_samples if sample not in ignore_samples]

    if len(valid_samples) < 2:
        # nothing to perturb
        return np.empty((0, X_sub.shape[1])), pd.DataFrame(
            columns=['sample', 'perturbation', 'idx', 'subsample_idx']
        )
    
    # 4) Loop over valid source→target pairs
    recons, metas = [], []
    for src in tqdm(valid_samples, desc="Perturbing Samples ..."):
        for tgt in valid_samples:
            if src == tgt:
                continue

            # Get the rows for the source and target samples
            src_rows = sample_to_idx[src]
            tgt_rows = sample_to_idx[tgt]
            # Randomly select subsamples from the source and target rows
            pick_src = rng.choice(src_rows, size=n_subsamples, replace=True)
            pick_tgt = rng.choice(tgt_rows, size=n_subsamples, replace=True)
            
            zs_perturb = []
            for branch in branch_names:
                tf.random.set_seed(seed) # re-seed for every inference
                if branch != sample_branch_name:
                    _x = X_sub[pick_src, :]
                else:
                    _x = X_sub[pick_tgt, :]
                zp = obj.encoders[branch](_x)          
                zs_perturb.append(obj.reparam_layers[branch](zp).numpy())

            y_src = y_sub[pick_src, :]

            with tf.device('/CPU:0'):
                recon = obj.decoder([y_src] + zs_perturb).numpy()

            recons.append(recon)
            perturb_meta = pd.DataFrame(
                data={
                    'reconstruction_type': 'perturbed',
                    'perturb_type': sample_col,
                    'source': [src]*n_subsamples,
                    'target': [tgt]*n_subsamples
                }
            )
            perturb_meta = pd.concat(
                [meta_sub.iloc[pick_src].reset_index(drop=True), perturb_meta],
                axis=1
            )
            if integrate_over_cell_types and 'cell_prop_type' in meta_sub.columns:
                perturb_meta['cell_prop_type'] = 'integrated'
            metas.append(perturb_meta)

    # 5) Collate and return
    x_reconst_sample_perturb = np.vstack(recons)
    sample_perturb_meta = pd.concat(metas, ignore_index=True)
    sample_perturb_meta.reset_index(drop=True, inplace=True)

    return x_reconst_sample_perturb, sample_perturb_meta