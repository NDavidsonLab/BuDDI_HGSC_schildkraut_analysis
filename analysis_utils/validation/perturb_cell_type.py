from typing import Iterable, Optional, Tuple, List, Any

from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from ..buddi4data import BuDDI4Data

def perturb_cell_type(
    obj: Any,
    data: BuDDI4Data,
    cell_type_col: str = 'cell_type',
    idx: Optional[Iterable[int]] = None,
    n_subsamples: int = 500,
    seed: Optional[int] = 42,
    ignore_cell_types: Iterable[str] = ['random'],
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Perturb dominant cell types in your data and get reconstructions.

    :param obj: BuDDI4 model object.
    :param data: BuDDI4Data object.
    :param cell_type_col: Column name in meta data for cell types.
    :param idx: Indices of the data to use for perturbation.
    :param n_subsamples: Number of subsamples to use for each perturbation.
    :param seed: Random seed for reproducibility.
    :param ignore_cell_types: Cell types to ignore during perturbation.
    :return: Tuple of reconstructed data and metadata.
        - x_reconst_ct_perturb: Reconstructed data after perturbation.
        - ct_perturb_meta: Metadata for the perturbation.
    """
    rng = np.random.default_rng(seed)

    # 1) Gather & subset
    X = data.get('kp', 'X')
    y = data.get('kp', 'Y')
    meta = data.get('kp', 'meta')
    if idx is None:
        idx = np.arange(X.shape[0])

    X_sub, y_sub = X[idx,:], y[idx,:]
    meta_sub = meta.iloc[idx].copy()

    # 2) Build concatenated z_sub
    branch_names = list(obj.encoder_branch_names)
    if 'slack' in obj.encoders:
        branch_names += ['slack']

    # 3) Map cell types to indices & ys
    ct_to_idx = {ct: np.where(meta_sub[cell_type_col] == ct)[0]
                 for ct in data.cell_type_names}
    valid_ct = [ct for ct, rows in ct_to_idx.items() if rows.size > 0]
    valid_ct = [ct for ct in valid_ct if ct not in ignore_cell_types]

    if len(valid_ct) < 2:
        # nothing to perturb
        return np.empty((0, X_sub.shape[1])), pd.DataFrame(
            columns=[cell_type_col, f"{cell_type_col}_perturb_target"]
        )

    #ct_to_y = {ct: y_sub[rows] for ct, rows in ct_to_idx.items()}

    # 4) Loop over valid source→target pairs
    recons, metas = [], []
    for src in tqdm(valid_ct, desc="Perturb CT"):
        for tgt in valid_ct:
            if src == tgt:
                continue

            src_rows = ct_to_idx[src]
            tgt_rows = ct_to_idx[tgt]
            pick_src = rng.choice(src_rows, size=n_subsamples, replace=True)
            pick_tgt = rng.choice(tgt_rows, size=n_subsamples, replace=True)

            zs_src = []
            for branch in branch_names:
                zp = obj.encoders[branch](X_sub[pick_src, :])
                zs_src.append(obj.reparam_layers[branch](zp).numpy())

            y_tgt = y_sub[pick_tgt, :]

            with tf.device('/CPU:0'):
                recon = obj.decoder([y_tgt] + zs_src).numpy()

            recons.append(recon)
            perturb_meta = pd.DataFrame(
                data={
                    'reconstruction_type': 'perturbed',
                    'perturb_type': cell_type_col,
                    'source': [src]*n_subsamples,
                    'target': [tgt]*n_subsamples
                }
            )
            perturb_meta = pd.concat(
                [meta_sub.iloc[pick_src].reset_index(drop=True), perturb_meta],
                axis=1
            )
            metas.append(perturb_meta)

    # 5) Collate and return
    x_reconst_ct_perturb = np.vstack(recons)
    ct_perturb_meta = pd.concat(metas, ignore_index=True)
    ct_perturb_meta.reset_index(drop=True, inplace=True)
    
    return x_reconst_ct_perturb, ct_perturb_meta