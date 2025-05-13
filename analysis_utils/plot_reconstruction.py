from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_reconstruction(
    x_reconstruct: np.ndarray,
    meta_reconstruct: pd.DataFrame,
    reduction_method: str = 'UMAP',
    color_by: List[str] = ['sample_id', 'stim'],
    palette: str = 'tab20'
):
    from sklearn.decomposition import PCA
    from umap.umap_ import UMAP
    
    fig, ax = plt.subplots(1, len(color_by), figsize=(5 * len(color_by), 5))

    if reduction_method == 'PCA':
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(x_reconstruct, ensure_all_finite=True)
    elif reduction_method == 'UMAP':
        pca_reducer = PCA(n_components=30)
        pca = pca_reducer.fit_transform(x_reconstruct)
        reducer = UMAP(n_components=2, random_state=42, n_jobs=1)
        coords = reducer.fit_transform(pca, ensure_all_finite=True)

    df_reduction = pd.DataFrame(
        coords, columns=[f'{reduction_method}1',f'{reduction_method}2'])
    df_reduction = pd.concat(
        [df_reduction, meta_reconstruct.reset_index(drop=True)], axis=1)
    df_reduction['dataset'] = 'perturb'
    
    for i, col in enumerate(color_by):
        # Plot the reconstructed data
        sns.scatterplot(
            data=df_reduction,
            x=f'{reduction_method}1',
            y=f'{reduction_method}2',
            hue=col,
            palette=palette,
            ax=ax[i],
            alpha=0.7
        )
        ax[i].set_title(f'Reconstructed Data {col}')
        if df_reduction[col].nunique() > 20:
            ax[i].legend([],[], frameon=False)
    
    plt.tight_layout()
    plt.show()