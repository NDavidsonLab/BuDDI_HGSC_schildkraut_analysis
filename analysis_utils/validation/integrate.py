from typing import Optional, List

import numpy as np
import pandas as pd

def _integrate_z(
    z_param: np.ndarray,
    group_idx: Optional[List[int]]=None
):
    """
    Integrate over the latent space z.

    :param z_param: Latent space parameter of shape (n_samples, 2 * z_dim).
        The first z_dim columns are the mean, and the next z_dim columns are the log variance.
    :param group_idx: Indices of the groups to integrate over.
    """

    n, two_z_dim = z_param.shape
    if group_idx is None:
        group_idx = [0] * n
    else:
        if len(group_idx) != n:
            raise ValueError("group_idx must have the same length as z_param.")
    group_idx = np.asarray(group_idx)

    z_dim = two_z_dim // 2
    if two_z_dim % 2 != 0:
        raise ValueError("z_param must have an even number of columns.")
    
    z_mus = z_param[:, :z_dim]
    z_log_vars = z_param[:, z_dim:]
    z_vars = np.exp(z_log_vars)

    mu_out = np.zeros_like(z_mus)
    logvar_out = np.zeros_like(z_log_vars)

    for g in np.unique(group_idx):
        mask = (group_idx == g)
        mus_g = z_mus[mask]
        vars_g = z_vars[mask]

        E_var = vars_g.mean(axis=0, keepdims=True)
        Var_mu = np.var(mus_g, axis=0, keepdims=True)
        marginal_var = E_var + Var_mu
        log_marginal = np.log(marginal_var)
        E_mu = mus_g.mean(axis=0, keepdims=True)

        mu_out[mask] = E_mu
        logvar_out[mask] = log_marginal

    return np.concatenate([mu_out, logvar_out], axis=1)

def integrate_z(
    z_param: np.ndarray,
    metadata: Optional[pd.DataFrame] = None,
    stratify_by: Optional[List[str]] = None,
):
    
    group_idx = _produce_group_idx(
        metadata=metadata,
        stratify_by=stratify_by,
    )

    return _integrate_z(z_param, group_idx)

def _integrate_y(
    y: np.ndarray,
    group_idx: Optional[List[int]] = None,
):
    """
    Integrate over the latent space y.

    :param y: Latent space parameter of shape (n_samples, n_classes).
    :param group_idx: Indices of the groups to integrate over.
    """
    if group_idx is None:
        group_idx = [0] * y.shape[0]
    else:
        if len(group_idx) != y.shape[0]:
            raise ValueError("group_idx must have the same length as y.")
    group_idx = np.asarray(group_idx)

    y_out = np.zeros_like(y)
    for g in np.unique(group_idx):
        mask = (group_idx == g)
        y_g = y[mask]
        y_out[mask] = y_g.mean(axis=0, keepdims=True)

    return y_out

def integrate_y(
    y: np.ndarray,
    metadata: Optional[pd.DataFrame] = None,
    stratify_by: Optional[List[str]] = None,
):
    
    group_idx = _produce_group_idx(
        metadata=metadata,
        stratify_by=stratify_by,
    )

    return _integrate_y(y, group_idx)

def _produce_group_idx(
    metadata: Optional[pd.DataFrame] = None,
    stratify_by: Optional[List[str]] = None,
):
    """
    Produce group indices for stratification.

    :param metadata: Metadata DataFrame.
    :param stratify_by: List of columns to stratify by.
    :return: Group indices.
    """
    group_idx = None
    if stratify_by is not None:
        if metadata is None:
            raise ValueError("metadata must be provided if stratify_by is specified.")
        if not all(col in metadata.columns for col in stratify_by):
            raise ValueError("All columns in stratify_by must be present in metadata.")    
        group_idx = metadata.groupby(stratify_by, sort=False).ngroup().to_numpy()

    return group_idx