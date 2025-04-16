from typing import List, Dict, Union, Optional, Callable

import numpy as np
import pandas as pd
import tensorflow as tf

def integrate_over(
    y: np.ndarray,
    metadata: Optional[pd.DataFrame] = None,
    stratify_by: Optional[str] = None,
    int_fn: Optional[Callable] = None,
) -> np.ndarray:
    """
    Integrate
    """

    if metadata is not None:
        if stratify_by is None:
            raise ValueError("Stratification column must be provided if metadata is provided.")
        else:
            if stratify_by not in metadata.columns:
                raise ValueError(f"Stratification column '{stratify_by}' not found in metadata.")
        stratify = True
    else:
        stratify = False

    if int_fn is None:
        def int_fn(x: np.ndarray) -> np.ndarray:
            return np.mean(x, axis=0, keepdims=True)
        
    if isinstance(y, tf.Tensor):
        y = y.numpy()
    elif not isinstance(y, np.ndarray):
        raise ValueError("Input y must be a numpy array or a TensorFlow tensor.")

    if not stratify:
        return int_fn(y)
    
    if len(y) != len(metadata):
        raise ValueError("The number of rows in matrix and metadata must match.")
        
    class_labels = metadata[stratify_by].values
    unique_categories = np.unique(class_labels)
    cat_int_dict = {}

    for cat in unique_categories:
        cat_idx = np.where(
            metadata[stratify_by] == cat
        )[0]
        cat_int_dict[cat] = int_fn(y[cat_idx, :])

    # Map each row to its group mean
    y_strat = np.vstack([cat_int_dict[label] for label in class_labels])

    return y_strat

def _int_z(
    z_param: np.ndarray,
    eps: float = 1e-8,
):
    if isinstance(z_param, tf.Tensor):
        z_param = z_param.numpy()
    elif not isinstance(z_param, np.ndarray):
        raise ValueError("Input z must be a numpy array or a TensorFlow tensor.")
    
    latent_dim = z_param.shape[1] // 2
    z_mu = z_param[:, :latent_dim]
    z_logvar = z_param[:, latent_dim:]
    _z_var = np.exp(z_logvar)

    emp_z_mu = np.mean(z_mu, axis=0)
    emp_var = np.mean(_z_var, axis=0)
    emp_logvar = np.log(emp_var + eps)

    return np.concatenate(
            [emp_z_mu[np.newaxis,:], emp_logvar[np.newaxis,:]], axis=1
        )    

def integrate_z(
    z_param,
    meta: Optional[pd.DataFrame] = None,
    category_col: Optional[str] = None,
    eps=1e-8,
):

    if isinstance(z_param, tf.Tensor):
        z_param = z_param.numpy()
    elif isinstance(z_param, pd.DataFrame):
        z_param = z_param.to_numpy()
    elif isinstance(z_param, np.ndarray):
        pass
    else:
        raise ValueError(f"Unsupported type {type(z_param)}")

    latent_dim = z_param.shape[1] // 2
    
    if meta is not None and category_col is not None:
        if category_col not in meta.columns:
            raise ValueError(f"Column {category_col} not found in meta DataFrame")
        
        categories = meta[category_col].unique()

        z_mu = z_param[:, :latent_dim]
        z_logvar = z_param[:, latent_dim:]

        emp_z_param_dict = {}
        for cat in categories:
            _idx = np.where(meta[category_col] == cat)[0]
            _z_mu = z_mu[_idx, :]
            _z_logvar = z_logvar[_idx, :]
            _z_var = np.exp(_z_logvar)

            # Compute empirical mean and variance for class specific latent space
            emp_z_mu = np.mean(_z_mu, axis=0)
            emp_var = np.mean(_z_var, axis=0)
            emp_logvar = np.log(emp_var + eps)

            emp_z_param_dict[cat] = np.concatenate(
                [emp_z_mu[np.newaxis,:], emp_logvar[np.newaxis,:]], axis=1
            )

        return emp_z_param_dict
    else:
        z_mu = z_param[:, :latent_dim]
        z_logvar = z_param[:, latent_dim:]
        _z_var = np.exp(z_logvar)

        emp_z_mu = np.mean(z_mu, axis=0)
        emp_var = np.mean(_z_var, axis=0)
        emp_logvar = np.log(emp_var + eps)

        return np.concatenate(
            [emp_z_mu[np.newaxis,:], emp_logvar[np.newaxis,:]], axis=1
        )        

def resample_z(
    z_params: Union[List[np.ndarray],Dict[str, np.ndarray]],
    resample_layer: Callable
):
    
    if isinstance(z_params, List):
        pass
    elif isinstance(z_params, Dict):
        z_params = [z_params[k] for k in z_params.keys()]

    return np.concatenate(
        [resample_layer(z_param).numpy()\
        for z_param in z_params],
        axis=1
    )

def label_condition(df, _ct_source, _ct_target):
    conditions = [
        (df['cell_type_perturb_target'] == '-') & (df['cell_type'] == _ct_source),
        (df['cell_type_perturb_target'] == '-') & (df['cell_type'] == _ct_target),
        (df['cell_type'] == _ct_source) & (df['cell_type_perturb_target'] == _ct_target)
    ]
    choices = ['source', 'target', 'perturb']

    df['perturbation'] = np.select(conditions, choices, default='-')
    return df