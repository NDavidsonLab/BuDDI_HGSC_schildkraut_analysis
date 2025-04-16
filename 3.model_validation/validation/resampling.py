from typing import Dict, List, Optional
from joblib import Parallel, delayed

import numpy as np
from scipy.stats import spearmanr, pearsonr
from tensorflow import keras
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.models import Model

class ResamplingDecoder(keras.Model):
    def __init__(
            self, 
            decoder: keras.Model,
            n_y: int,
            z_shape: Dict[str, int],
            z_order: Optional[List[str]] = None,
            ## TODO replace this with in-place instantiation
            _reparam_layer: Optional[keras.layers.Layer] = None,
        ):

        super(ResamplingDecoder, self).__init__()        
        
        if z_order is None:
            z_order = z_shape.keys()
        
        _inputs = [Input(shape=(n_y,), name='y')]
        _z_inputs = []
        for z_key in z_order:
            _z_inputs.append(Input(shape=(z_shape[z_key] * 2,), name=z_key))
        _z_resample = [
            _reparam_layer(_z_input) for _z_input in _z_inputs
        ]
        _concat = Concatenate()(_inputs + _z_resample)
        _decoded = decoder(_concat)

        self.__decoder = Model(
            inputs=_inputs + _z_inputs,
            outputs=_decoded,
            name='resampling_decoder'
        )

    def call(self, inputs):
        return self.__decoder(inputs)
    
def resample(
    decoder,
    z_params,
    y,
    n_resamples: int,    
    _reparam_layer: Optional[keras.layers.Layer] = None,
):

    return None    
    
def compute_reconstruction_corr(
    x_resample,
    x_truth,
    n_resamples = None,
    method = 'pearson',
    n_jobs=-1,
):

    n, m = x_truth.shape    

    if n_resamples is None:
        n_resamples = x_resample.shape[0] // n

    correlations = np.zeros((n, n_resamples))

    def compute_row_corr(i, j):
        x_row = x_resample[j * n + i]
        y_row = x_truth[i]
        if method == 'spearman':
            corr, _ = spearmanr(x_row, y_row)
        elif method == 'pearson':
            corr, _ = pearsonr(x_row, y_row)
        else:
            raise ValueError("method must be 'spearman' or 'pearson'")
        return i, j, corr
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_row_corr)(i, j) for j in range(n_resamples) for i in range(n)
    )

    for i, j, corr in results:
        correlations[i, j] = corr

    return correlations