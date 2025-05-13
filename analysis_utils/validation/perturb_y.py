from typing import Iterable, Any, Optional, List

from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf

from .integrate import integrate_z

def _perturb_y(
    obj: Any,
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    cell_type_names: List[str],
    cell_type_col: str,
    idx: Optional[Iterable[int]] = None,
    n_subsamples: int = 500,
    n_resamples: int = 1,
    integrate_over: Optional[List[str]] = None,
    ignore_cell_types: Iterable[str] = [],
    seed: Optional[int] = 42,
):
    """
    Perturb the latent space of a model by sampling from the encoder and decoder.

    :param obj: BuDDI4 model object.
    :param X: Input normalized expression matrix of shape (n_samples, n_genes).
    :param y: Input cell type proportions matrix of shape (n_samples, n_cell_types).
        rows should sum up to 1.
    :param meta: Metadata associated with the data of shape (n_samples, n_meta_features). 
        Must contain column for cell types.  
    :param cell_type_names: List of cell type names.
    :param cell_type_col: Column name in metadata for cell types.
    :param idx: Indices of input data to use for perturbation. If None, all samples are used.
    :param n_subsamples: Number of subsamples to draw from each source cell type.
    :param n_resamples: Number of resamples to draw from each source cell type.
    :param integrate_over: List of latent spaces to integrate over.
        If None, no integration is performed.
    :param ignore_cell_types: List of cell types to ignore during perturbation.
    :param seed: Random seed for reproducibility.
    """
    
    rng = np.random.default_rng(seed)

    if idx is None:
        idx = np.arange(X.shape[0])

    X_sub, y_sub = X[idx,:], y[idx,:]
    meta_sub = meta.iloc[idx].copy()

    # 1) Integrateover z if needed
    branch_names = list(obj.encoder_branch_names)
    if 'slack' in obj.encoders:
        branch_names += ['slack']

    z_param_int = {}
    if integrate_over is not None:
        if not all(branch_name in branch_names for branch_name in integrate_over):
            raise ValueError(f"Branches {integrate_over} not found in encoder branches.")

        for branch_name in integrate_over:
            z_param_int[branch_name] = integrate_z(
                obj.encoders[branch_name](X_sub).numpy(),
            )
    else:
        integrate_over = []

    # 2) Map cell types to indices & ys
    ct_to_idx = {ct: np.where(meta_sub[cell_type_col] == ct)[0]
                 for ct in cell_type_names}
    valid_ct = [ct for ct, rows in ct_to_idx.items() if rows.size > 0]
    valid_ct = [ct for ct in valid_ct if ct not in ignore_cell_types]

    if len(valid_ct) < 2:
        # nothing to perturb
        return np.empty((0, X_sub.shape[1])), pd.DataFrame(
            columns=[cell_type_col, f"{cell_type_col}_perturb_target"]
        )
    
    # 3) Loop over valid source→target pairs
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
                if branch in integrate_over:
                    zp = z_param_int[branch][pick_src]
                else:
                    zp = obj.encoders[branch](X_sub[pick_src, :])

                if n_resamples > 1:
                    zs_resample = []
                    for _ in range(n_resamples):
                        tf.random.set_seed(seed) # re-seed for every inference
                        zs_resample.append(
                            obj.reparam_layers[branch](zp).numpy()
                        )                
                    zs_src.append(np.vstack(zs_resample))
                else:
                    zs_src.append(
                        obj.reparam_layers[branch](zp).numpy()
                    )

            y_tgt = y_sub[pick_tgt, :]
            y_tgt = np.tile(y_tgt, (n_resamples, 1))

            with tf.device('/CPU:0'):
                recon = obj.decoder([y_tgt] + zs_src).numpy()

            recons.append(recon)
            perturb_meta = pd.DataFrame(
                data={
                    'reconstruction_type': 'perturbed',
                    'perturb_type': cell_type_col,
                    'source': [src]*n_subsamples*n_resamples,
                    'target': [tgt]*n_subsamples*n_resamples
                }
            )
            perturb_meta = pd.concat(
                [meta_sub.iloc[pick_src].reset_index(drop=True), perturb_meta],
                axis=1
            )
            metas.append(perturb_meta)

    # 4) Collate and return
    x_reconst_ct_perturb = np.vstack(recons)
    ct_perturb_meta = pd.concat(metas, ignore_index=True)
    ct_perturb_meta.reset_index(drop=True, inplace=True)
    
    return x_reconst_ct_perturb, ct_perturb_meta