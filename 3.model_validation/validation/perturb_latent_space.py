from typing import Optional, Iterable, Callable, Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model

from .utils import integrate_z

def perturb_z_branch(
    decoder: Model,
    z_param_dict: Dict[str, np.ndarray],
    z_to_perturb: str,
    y: np.ndarray,
    int_over_y: bool,
    int_over_z: bool,
    metadata: pd.DataFrame,
    idx: Optional[Iterable] = None,
    n_resamples: int = 1,
    # TODO: make this type hint the reparam layer class
    _reparam_layer: Optional[Callable] = None,
): 
    """
    """

    if z_to_perturb not in z_param_dict:
        raise ValueError(f"z_param_dict does not contain {z_to_perturb}")
    if z_to_perturb not in metadata.columns:
        raise ValueError(f"metadata does not contain {z_to_perturb}")

    y = y.numpy() if isinstance(y, tf.Tensor) else y

    if idx is None:
        idx = np.arange(y.shape[0])
    metadata = metadata.iloc[idx,:].copy().loc[:, [z_to_perturb]]
    # tile metadata to match the number of resamples
    metadata = pd.concat(
        [metadata] * n_resamples,
        axis=0,
    )

    z_param_dict = z_param_dict.copy()
    for z_name, z_param in z_param_dict.items():
        if isinstance(z_param, tf.Tensor):
            z_param = z_param.numpy()

        if z_name == z_to_perturb or not int_over_z:
            # preserve the z_param that we are perturbing          
            z_param_dict[z_name] = np.tile(
                z_param[idx,:],
                (n_resamples, 1)
            )
        else:
            # integrate over the entire z_param and tile to match the perturbed z_param and resampled y
            z_param_dict[z_name] = np.tile(
                integrate_z(z_param),
                (len(idx) * n_resamples, 1)
            )
    
    if int_over_y:
        # if int_over_y is True
        # we integrate over y
        y = np.tile(
            np.mean(y, axis=0, keepdims=True),
            (len(idx) * n_resamples, 1)
        )
    else:
        # if int_over_y is False
        # we subsample y to match the dimensions of z_params
        y = np.tile(
            y[idx,:],
            (n_resamples, 1)
        )

    perturb_categories = list(metadata[z_to_perturb].unique())
    xs_perturb = []
    metas_perturb = []
    for _perturb_source in perturb_categories:
        for _perturb_target in perturb_categories:
            if _perturb_source == _perturb_target:
                continue

            _source_idx = np.where(metadata[z_to_perturb] == _perturb_source)[0]
            _target_idx = np.where(metadata[z_to_perturb] == _perturb_target)[0]
            # if the number of source and target indices are not equal
            # we randomly sample the target indices to match the source indices
            if len(_target_idx) != len(_source_idx):
                _target_idx = np.random.choice(
                    _target_idx, 
                    size=len(_source_idx), 
                    replace=len(_target_idx) < len(_source_idx)
                )

            z_perturb = []

            for z_name, z_param in z_param_dict.items():
                if z_name == z_to_perturb:
                    _z_param = z_param[_target_idx, :]
                else:
                    _z_param = z_param[_source_idx, :]

                z_perturb.append(
                    _reparam_layer(
                        _z_param
                    ).numpy()
                )

            with tf.device('/CPU:0'):
                xs_perturb.append(
                    decoder(
                        np.concatenate(
                            [y[_source_idx,:]] + z_perturb,
                            axis=1
                        )
                    ).numpy()
                )
            metas_perturb.append(
                pd.DataFrame(
                    data = {
                        z_to_perturb: [_perturb_source] * len(_source_idx),
                        f'{z_to_perturb}_target': [_perturb_target] * len(_source_idx),
                    }
                )
            )
    
    x_perturb = np.concatenate(xs_perturb, axis=0)
    meta_perturb = pd.concat(metas_perturb, axis=0)
    meta_perturb.reset_index(drop=True, inplace=True)

    return x_perturb, meta_perturb

def perturb_y_branch(
    decoder: Model,
    z_param_dict: Dict[str, np.ndarray],
    y_col: str,
    y: np.ndarray,
    int_over_z: bool,
    metadata: pd.DataFrame,
    idx: Optional[Iterable] = None,
    n_resamples: int = 1,
    _reparam_layer: Optional[Callable] = None,
):  
    """
    """

    if y_col not in metadata.columns:
        raise ValueError(f"metadata does not contain {y_col}")

    y = y.numpy() if isinstance(y, tf.Tensor) else y

    if idx is None:
        idx = np.arange(y.shape[0])
    metadata = metadata.iloc[idx, :].copy().loc[:, [y_col]]

    # Tile metadata
    metadata = pd.concat([metadata] * n_resamples, axis=0)

    z_param_dict = z_param_dict.copy()
    for z_name, z_param in z_param_dict.items():
        if isinstance(z_param, tf.Tensor):
            z_param = z_param.numpy()

        if int_over_z:
            z_param_dict[z_name] = np.tile(
                integrate_z(z_param),
                (len(idx) * n_resamples, 1)
            )
        else:
            z_param_dict[z_name] = np.tile(
                z_param[idx, :],
                (n_resamples, 1)
            )

    # Synthetic 1-hot y classes
    class_ids = list(metadata[y_col].unique())
    num_classes = len(class_ids)
    y = y[idx, :]
    y = np.tile(y, (n_resamples, 1))

    xs_perturb = []
    metas_perturb = []

    for source_class in class_ids:
        for target_class in class_ids:
            if source_class == target_class:
                continue

            source_idx = np.where(metadata[y_col] == source_class)[0]
            target_idx = np.where(metadata[y_col] == target_class)[0]

            if len(target_idx) == 0 or len(source_idx) == 0:
                continue

            if len(target_idx) != len(source_idx):
                target_idx = np.random.choice(
                    target_idx,
                    size=len(source_idx),
                    replace=len(target_idx) < len(source_idx)
                )

            # Construct synthetic 1-hot y
            y_perturb = np.zeros_like(y[source_idx, :])
            y_perturb[:, target_class] = 1.0

            z_perturb = []
            for z_name, z_param in z_param_dict.items():
                z_perturb.append(
                    _reparam_layer(z_param[source_idx, :]).numpy()
                )

            with tf.device('/CPU:0'):
                xs_perturb.append(
                    decoder(
                        np.concatenate(
                            [y_perturb] + z_perturb,
                            axis=1
                        )
                    ).numpy()
                )

            metas_perturb.append(
                pd.DataFrame({
                    y_col: [source_class] * len(source_idx),
                    f"{y_col}_target": [target_class] * len(source_idx)
                })
            )

    x_perturb = np.concatenate(xs_perturb, axis=0)
    meta_perturb = pd.concat(metas_perturb, axis=0)
    meta_perturb.reset_index(drop=True, inplace=True)

    return x_perturb, meta_perturb

def impute_sc(
    decoder: Model,
    z_param_dict: Dict[str, np.ndarray],
    cell_type_names: List[str],
    int_over_z: bool,
    skip_int_z: Optional[List[str]]=None,
    metadata: Optional[pd.DataFrame]=None,
    idx: Optional[Iterable] = None,
    n_resamples: int = 1,
    # TODO: make this type hint the reparam layer class
    _reparam_layer: Optional[Callable] = None,
):
    
    if idx is None:
        idx = np.arange(next(iter(z_param_dict.values())).shape[0])

    if metadata is not None:
        metadata = metadata.iloc[idx,:].copy()
        # tile metadata to match the number of resamples
        metadata = pd.concat(
            [metadata] * n_resamples * len(cell_type_names),
            axis=0,
        )

    if skip_int_z is None:
        skip_int_z = []

    z_param_dict = z_param_dict.copy()
    for z_name, z_param in z_param_dict.items():
        if isinstance(z_param, tf.Tensor):
            z_param = z_param.numpy()[idx,:]
        if int_over_z and z_name not in skip_int_z:
            z_param_dict[z_name] = np.tile(
                integrate_z(z_param),
                (len(idx) * n_resamples, 1)
            )
        else:
            z_param_dict[z_name] = np.tile(
                z_param,
                (n_resamples, 1)
            )

    xs_perturb = []
    metas_perturb = []
    for i, cell_type_name in enumerate(cell_type_names):
        
        _y = np.zeros((
            len(idx) * n_resamples, 
            len(cell_type_names)))
        
        _y[:, i] = 1.0

        z_samp = []
        for z_name, z_param in z_param_dict.items():
            z_samp.append(
                _reparam_layer(
                    z_param
                ).numpy()
            )

        with tf.device('/CPU:0'):
            xs_perturb.append(decoder(
                np.concatenate(
                    [_y] + z_samp,
                    axis=1
                )
            ).numpy())

        metas_perturb.append(
            pd.DataFrame(
                data = {
                    'cell_type': [cell_type_name] * len(idx) * n_resamples,
                    'cell_prop_type': ['imputed'] * len(idx) * n_resamples,
                }
            )
        )

    x_perturb = np.concatenate(xs_perturb, axis=0)
    meta_perturb = pd.concat(metas_perturb, axis=0)
    meta_perturb.reset_index(drop=True, inplace=True)
    if metadata is not None:

        _meta = metadata.reset_index(drop=True, inplace=False).copy()
        _meta.drop(columns=meta_perturb.columns, inplace=True, errors='ignore')

        meta_perturb = pd.concat(
            [_meta, meta_perturb],
            axis=1
        )

    return x_perturb, meta_perturb