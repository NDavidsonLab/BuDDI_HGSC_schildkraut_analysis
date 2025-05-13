from typing import Iterable, Any, Optional, List
import warnings

from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf

from .integrate import integrate_z, integrate_y

def _perturb_z(
    obj: Any,
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    z_col_name: str,
    z_branch_name: str,
    idx: Optional[Iterable[int]] = None,
    n_subsamples: Optional[int] = None,
    n_resamples: Optional[int] = 1,
    integrate_over_y: bool = True,
    integrate_over: Optional[List[str]] = None,
    ignore_class: Iterable[str] = [],
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
    :param z_col_name: Column name in metadata for the latent space to be perturbed.
    :param z_branch_name: Branch name in the encoder for the latent space to be perturbed.
    :param idx: Indices of input data to use for perturbation. If None, all samples are used.
    :param n_subsamples: Number of subsamples to draw from each source cell type.
    :param n_resamples: Number of resamples to draw from each source cell type.
    :param integrate_over_y: Whether to integrate over y.
    :param integrate_over: List of latent spaces to integrate over.
        If None, no integration is performed. Should not include the branch to be perturbed.
    :param ignore_class: List of cell types to ignore during perturbation.
    :param seed: Random seed for reproducibility.
    :return: Tuple of reconstructed data and metadata.
        - x_reconst_sample_perturb: Reconstructed data after perturbation.
        - sample_perturb_meta: Metadata for the perturbation.
    """
    
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.RandomState(seed)

    if idx is None:
        idx = np.arange(X.shape[0])

    X_sub, y_sub = X[idx,:], y[idx,:]
    meta_sub = meta.iloc[idx].copy()

    # 1) Integrate over y if needed
    if integrate_over_y:
        y_sub = integrate_y(y_sub)

    # 2) Integrateover z if needed
    branch_names = list(obj.encoder_branch_names)
    if 'slack' in obj.encoders:
        branch_names += ['slack'] 

    z_param_int = {}
    if integrate_over is not None:
        if not all(branch_name in branch_names for branch_name in integrate_over):
            raise ValueError(f"Branches {integrate_over} not found in encoder branches.")
        
        if z_branch_name in integrate_over:
            raise ValueError(f"Branch {z_branch_name} to be perturbed cannot be integrated over.")

        for branch_name in integrate_over:
            z_param_int[branch_name] = integrate_z(
                obj.encoders[branch_name](X_sub).numpy(),
            )
    else:
        integrate_over = []

    # 3) Map samples to indices & ys
    perturb_class_names = meta_sub[z_col_name].unique()
    class_to_idx = {sample: np.where(meta_sub[z_col_name] == sample)[0]
                     for sample in perturb_class_names}
    valid_class_names = [_class for _class, rows in class_to_idx.items() if rows.size > 0]
    valid_class_names = [_class for _class in valid_class_names if _class not in ignore_class]

    if len(valid_class_names) < 2:
        # nothing to perturb
        return np.empty((0, X_sub.shape[1])), pd.DataFrame(
            columns=['sample', 'perturbation', 'idx', 'subsample_idx']
        )
    
    # 4) Loop over valid source→target pairs
    if n_resamples is None:
        n_resamples = 1
    recons, metas = [], []
    for src in tqdm(valid_class_names, desc=f"Perturbing {z_branch_name} ..."):
        for tgt in valid_class_names:
            if src == tgt:
                continue

            # Get the rows for the source and target samples
            src_rows = class_to_idx[src]
            tgt_rows = class_to_idx[tgt]
            # Randomly select subsamples from the source and target rows
            if n_subsamples is None:
                if len(src_rows) != len(tgt_rows):
                    n = min(len(src_rows), len(tgt_rows))
                    warnings.warn(
                        f"Source ({src}) and target ({tgt}) cell types have different number of samples. "
                        f"Subsampling to the minimum number of samples: ({n})."
                    )
                    if len(src_rows) != n:
                        pick_src = rng.choice(src_rows, size=n, replace=False)
                    if len(tgt_rows) != n:
                        pick_tgt = rng.choice(tgt_rows, size=n, replace=False)
                else:                    
                    pick_src = src_rows
                    pick_tgt = tgt_rows
            else:
                pick_src = rng.choice(src_rows, size=n_subsamples, replace=True)
                pick_tgt = rng.choice(tgt_rows, size=n_subsamples, replace=True)
            
            zs_perturb = []
            for branch in branch_names:
                if branch in integrate_over:
                    zp = z_param_int[branch][pick_src]
                else:
                    if branch != z_branch_name:
                        _x = X_sub[pick_src, :]
                    else:
                        _x = X_sub[pick_tgt, :]
                    zp = obj.encoders[branch](_x)
                if n_resamples > 1:
                    zs_resample = []
                    for _ in range(n_resamples):
                        tf.random.set_seed(seed) # re-seed for every inference
                        zs_resample.append(
                            obj.reparam_layers[branch](zp).numpy()
                        )                
                    zs_perturb.append(np.vstack(zs_resample))
                else:
                    zs_perturb.append(
                        obj.reparam_layers[branch](zp).numpy()
                    )

            y_src = y_sub[pick_src, :]
            y_src = np.tile(y_src, (n_resamples, 1))

            with tf.device('/CPU:0'):
                recon = obj.decoder([y_src] + zs_perturb).numpy()

            recons.append(recon)
            _n = recon.shape[0]
            perturb_meta = pd.DataFrame(
                data={
                    'reconstruction_type': 'perturbed',
                    'perturb_type': z_col_name,
                    'source': [src]*_n,
                    'target': [tgt]*_n
                }
            )
            orig_meta = pd.concat(
                [meta_sub.iloc[pick_src].reset_index(drop=True)] * n_resamples,
                axis=0
            ).reset_index(drop=True)
            perturb_meta = pd.concat(
                [orig_meta, perturb_meta],
                axis=1
            )

            if integrate_over_y and 'cell_prop_type' in meta_sub.columns:
                perturb_meta['cell_prop_type'] = 'integrated'
            metas.append(perturb_meta)

    # 5) Collate and return
    x_reconst_perturb = np.vstack(recons)
    perturb_meta = pd.concat(metas, ignore_index=True)
    perturb_meta.reset_index(drop=True, inplace=True)

    return x_reconst_perturb, perturb_meta