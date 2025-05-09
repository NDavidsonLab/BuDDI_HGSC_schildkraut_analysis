from typing import Iterable, Optional, Tuple, Any
import numpy as np
import pandas as pd
import tensorflow as tf

from ..buddi4data import BuDDI4Data

def reconstruct(
    obj: Any,
    data: BuDDI4Data,
    idx: Optional[Iterable[int]] = None,
    n_resamples: int = 1,
    seed: Optional[int] = 42,
    random_uniform_y: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Perform raw reconstruction (no perturbation), optionally repeating
    the stochastic reparameterization `n_resamples` times, and
    optionally substituting Y for a random uniform softmax per row.

    Parameters
    ----------
    obj
        Model object exposing:
          - encoder_branch_names: List[str]
          - encoders: Dict[str, tf.keras.Layer]
          - reparam_layers: Dict[str, tf.keras.Layer]
          - decoder: callable accepting [y, z1, z2, …]
    data
        BuDDI4Data exposing:
          - get('kp','X') -> np.ndarray
          - get('kp','Y') -> np.ndarray
          - get('kp','meta') -> pd.DataFrame
    idx
        Optional indices to subset the observations before encoding.
    n_resamples
        How many times to sample from each z‐distribution (default: 1).
    seed
        RNG seed for reproducibility of both NumPy and TF sampling.
    random_uniform_y
        If True, generate a new random uniform Y matrix (softmax‐normalized
        per row) for each resample; otherwise reuse the original Y.

    Returns
    -------
    x_recon : np.ndarray
        Stacked reconstructions, shape = (n_resamples * n_obs, x_dim).
    meta_recon : pd.DataFrame
        Original metadata repeated for each resample, with columns added:
          reconstruction_type='reconstruction'
          perturb_type='unperturbed'
          source=None
          target=None
    """
    # set seeds
    rng = np.random.default_rng(seed)
    if seed is not None:
        tf.random.set_seed(seed)

    # 1) Pull & subset X, Y, meta
    X = data.get('kp', 'X')
    Y = data.get('kp', 'Y')
    meta = data.get('kp', 'meta')

    if idx is None:
        idx = np.arange(X.shape[0])

    X_sub = X[idx]
    y_sub = Y[idx]
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

    return x_recon, meta_recon