from typing import Iterable, Any, Optional, List
import warnings

from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf

from ..validation.integrate import integrate_z

def _impute_sc_expression(
    obj: Any,
    X: np.ndarray,
    meta: pd.DataFrame,
    cell_type_names: List[str],
    cell_type_col: Optional[str] = None,
    idx: Optional[Iterable[int]] = None,
    n_subsamples: Optional[int] = None,
    n_resamples: int = 10,
    integrate_over: Optional[List[str]] = None,
    ignore_cell_types: Iterable[str] = [],
    seed: Optional[int] = 42,
):    
    """
    """

    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.RandomState(seed)

    if idx is None:
        idx = np.arange(X.shape[0])
        
    X_sub = X[idx,:]
    meta_sub = meta.iloc[idx].copy()
    n_y = len(cell_type_names)

    if n_subsamples is not None:
        if n_subsamples > X_sub.shape[0]:
            raise ValueError("Number of subsamples cannot be greater than number of samples.")
        if n_subsamples < 1:
            raise ValueError("Number of subsamples must be at least 1.")
        idx = rng.choice(X_sub.shape[0], size=n_subsamples, replace=False)
        X_sub = X_sub[idx,:]
        meta_sub = meta_sub.iloc[idx].copy()

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

    # 2) Create single cell type proportions
    ct_to_y = {}
    for i, ct in enumerate(cell_type_names):
        if ct in ignore_cell_types:
            continue
        ct_to_y[ct] = np.zeros((X_sub.shape[0], n_y))
        ct_to_y[ct][:, i] = 1
        ct_to_y[ct] = ct_to_y[ct].astype(np.float32)

    if len(ct_to_y) == 0:
        raise ValueError("No valid cell types found in the provided cell type names.")        

    # 3) Loop over all cell types and impute 
    recons, metas = [], []
    for ct, y in tqdm(ct_to_y.items(), desc='Imputing single cell types'):
        if ct in ignore_cell_types:
            continue
        
        zs_src = []
        for branch in branch_names:
            if branch in integrate_over:
                zp = z_param_int[branch]
            else:
                zp = obj.encoders[branch](X_sub)

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

        _y = np.tile(y, (n_resamples, 1))

        with tf.device('/CPU:0'):
            recon = obj.decoder([_y] + zs_src).numpy()

        recons.append(recon)
        _n = recon.shape[0]

        meta_tile = pd.concat(
            [meta_sub] * n_resamples,
            axis=0,
        ).reset_index(drop=True)

        if cell_type_col is not None:
            source = meta_tile[cell_type_col].values
        else:
            source = ['-']*_n

        perturb_meta = pd.DataFrame(
            data={
                'reconstruction_type': 'impute_sc',
                'perturb_type': cell_type_col if cell_type_col is not None else 'cell_type',
                'source': source,
                'target': [ct]*_n
            }
        )
        perturb_meta = pd.concat(
            [meta_tile, perturb_meta],
            axis=1
        )
        metas.append(perturb_meta)

    # 4) Collate and return
    x_reconst_ct_perturb = np.vstack(recons)
    ct_perturb_meta = pd.concat(metas, ignore_index=True)
    ct_perturb_meta.reset_index(drop=True, inplace=True)
    
    return x_reconst_ct_perturb, ct_perturb_meta