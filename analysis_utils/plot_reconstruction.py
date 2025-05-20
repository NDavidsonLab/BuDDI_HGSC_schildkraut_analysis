from typing import List, Optional
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_reconstruction(
    x_reconstruct: np.ndarray,
    meta_reconstruct: pd.DataFrame,
    reduction_method: str = 'UMAP',
    color_by: List[str] = ['sample_id', 'stim'],
    palette: str = 'tab20',
    palette_continuous: str = 'viridis',
    ncols: int = 4,
    show_plot: bool = True,
    save_path: Optional[str] = None,
):
    from sklearn.decomposition import PCA
    from umap.umap_ import UMAP

    if ncols > len(color_by):
        ncols = len(color_by)
    nrows = math.ceil(len(color_by) / ncols)
    
    fig, ax = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if len(color_by) == 1:
        axes = [ax]
    axes = ax.flatten()

    if reduction_method == 'PCA':
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(x_reconstruct)
    elif reduction_method == 'UMAP':
        pca_reducer = PCA(n_components=30)
        pca = pca_reducer.fit_transform(x_reconstruct)
        reducer = UMAP(n_components=2, random_state=42, n_jobs=1)
        coords = reducer.fit_transform(pca, ensure_all_finite=True)
    else:
        raise ValueError("Invalid reduction method. Choose 'PCA' or 'UMAP'.")

    df_reduction = pd.DataFrame(
        coords, columns=[f'{reduction_method}1',f'{reduction_method}2'])
    df_reduction = pd.concat(
        [df_reduction, meta_reconstruct.reset_index(drop=True)], axis=1)
    df_reduction['dataset'] = 'perturb'
    
    for i, col in enumerate(color_by):

        # Plot the reconstructed data
        hue_is_numeric = pd.api.types.is_numeric_dtype(df_reduction[col])
        if hue_is_numeric:
            # If the hue is numeric, use a continuous palette
            _palette = palette_continuous
        else:
            # If the hue is categorical, use a discrete palette
            _palette = palette
        
        sns.scatterplot(
            data=df_reduction,
            x=f'{reduction_method}1',
            y=f'{reduction_method}2',
            hue=col,
            palette=_palette,
            ax=axes[i],
            alpha=0.7
        )
        axes[i].set_title(f'Reconstructed Data {col}')
        if (not hue_is_numeric) and df_reduction[col].nunique() > 20:
            axes[i].legend([],[], frameon=False)
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # print(f"Plot saved to {save_path}")
    
    if show_plot:
        plt.show()

    plt.close(fig)