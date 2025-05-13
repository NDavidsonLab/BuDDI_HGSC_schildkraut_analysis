from typing import Iterable, Optional, Tuple, Any
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

def reconstruct(
    obj: Any,
    data: Any,
    n_subsamples: Optional[int] = None,
    n_resamples: int = 1,
    seed: Optional[int] = 42,
    random_uniform_y: bool = False,
    query_kwargs: Optional[dict] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Function for BuDDI4 reconstruction (no perturbation) interfacing with BuDDI4Data object.
    Accepts optional query_kwargs to filter the data.
    Optionally repeating the stochastic reparameterization `n_resamples` times, and
    Optionally substituting Y for a random uniform softmax per row.

    :param obj: BuDDI4 model object.
    :param data: BuDDI4Data object.
    :param n_subsamples: Number of subsamples to draw from each source cell type.
    :param n_resamples: Number of resamples to perform.
    :param seed: Random seed for reproducibility.
    :param random_uniform_y: Whether to use a random uniform softmax for Y.
    :return: Tuple of reconstructed data and metadata.
        - x_recon: Reconstructed data.
        - meta_recon: Metadata for the reconstruction.
    """

    # 1) Pull & subset X, Y, meta
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
    n_samples = n_subsamples
    if n_samples is None:
        replace=False
    else:
        replace = True if len(data_query) < n_subsamples else False
    try:
        (X, y, meta) = data_query.get(
            ('X', 'Y', 'meta'),
            n_samples=n_samples,
            replace=replace,
            random_state=seed,
        )
    except Exception as e:
        raise ValueError(f"Error retrieving data from the BuDDI4Data object: {e}")

    return _reconstruct(
        obj=obj,
        X=X,
        y=y,
        meta=meta,
        idx=None,
        n_resamples=n_resamples,
        random_uniform_y=random_uniform_y,
        seed=seed,
    )

def _reconstruct(
    obj: Any,
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    idx: Optional[Iterable[int]] = None,
    n_resamples: int = 1,
    random_uniform_y: bool = False,
    seed: Optional[int] = 42,
):
    
    """
    Helper function for generalized reconstruction task (without perturbation), optionally repeating
    the stochastic reparameterization `n_resamples` times, and
    optionally substituting Y for a random uniform softmax per row.

    Always reset the reparameterization to stochastic at the end.

    :param obj: BuDDI4 model object.
    :param X: Input normalized expression matrix of shape (n_samples, n_genes).
    :param y: Input cell type proportions matrix of shape (n_samples, n_cell_types).
        rows should sum up to 1.
    :param meta: Metadata associated with the data of shape (n_samples, n_meta_features).
        Must contain column for cell types.
    :param idx: Indices of input data to use for reconstruction. If None, all samples are used.
    :param n_resamples: Number of resamples to perform.
    :param random_uniform_y: Whether to use a random uniform softmax for Y.
    :param seed: Random seed for reproducibility.
    :return: Tuple of reconstructed data and metadata.
        - x_recon: Reconstructed data.
        - meta_recon: Metadata for the reconstruction.
    """

    # set seeds
    if seed is not None:
        rng = np.random.default_rng(seed)
        # configure reparameterization
        # to be deterministic
        obj.set_reparam_deterministic(
            deterministic=True,
            seed=seed,
        )
    else:
        rng = np.random.default_rng()

    if idx is None:
        idx = np.arange(X.shape[0])

    # 1) Subset data by index if provided
    X_sub = X[idx]
    y_sub = y[idx]
    meta_sub = meta.iloc[idx].reset_index(drop=True).copy()

    # 2) Precompute z_params per branch
    branch_names = list(obj.encoder_branch_names)
    if 'slack' in obj.encoders:
        branch_names += ['slack']

    z_params = {
        branch: obj.encoders[branch](X_sub)
        for branch in branch_names
    }

    recons = []
    metas = []

    # 3) Repeat reparameterization + decode
    for _ in range(n_resamples):
        # sample zs
        tf.random.set_seed(seed) # re-seed for every inference
        zs = [
            obj.reparam_layers[branch](z_params[branch]).numpy()
            for branch in branch_names
        ]

        # prepare Y input
        if random_uniform_y:
            y_rand = rng.random(y_sub.shape)
            # softmax per row
            y_rand = np.exp(y_rand) / np.sum(np.exp(y_rand), axis=1, keepdims=True)
            y_input = y_rand
        else:
            y_input = y_sub

        # decode on CPU
        with tf.device('/CPU:0'):
            recon = obj.decoder([y_input] + zs).numpy()

        recons.append(recon)

        # assemble metadata
        meta_r = meta_sub.copy()
        meta_r['reconstruction_type'] = 'reconstruction'
        meta_r['perturb_type'] = 'unperturbed'
        meta_r['source'] = None
        meta_r['target'] = None
        metas.append(meta_r)

    # 4) Collate outputs
    x_recon = np.vstack(recons)
    meta_recon = pd.concat(metas, ignore_index=True)

    # reset reparameterization to stochastic
    obj.set_reparam_deterministic(
        deterministic=False,
        seed=None,
    )

    return x_recon, meta_recon