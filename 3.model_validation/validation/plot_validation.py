import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import umap.umap_ as umap

from .utils import label_condition

def plot_resampled_latent_space(
    x,
    metadata,
    color_by,
    panel_width=5,
    use_umap=False,
    show_plot=True,
    save_path=None,
):

    pca = PCA(n_components=50 if use_umap else 2)
    proj_pca = pca.fit_transform(x)

    if use_umap:
        _umap = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        proj_umap = _umap.fit_transform(proj_pca)

        proj_df = pd.DataFrame(
            data=proj_umap,
            columns=['UMAP1', 'UMAP2']
        )
        proj_df['PC1'] = proj_pca[:, 0]
        proj_df['PC2'] = proj_pca[:, 1]
    else:
        proj_df = pd.DataFrame(
            data=proj_pca,
            columns=['PC1', 'PC2']
        )

    _color_by = []
    for col in color_by:
        if col in metadata.columns:
            proj_df[col] = metadata[col].values
            _color_by.append(col)

    if len(_color_by) == 0:
        print("No columns found in metadata to color by.")
        return
    
    n_rows = 2 if use_umap else 1 # 1 row for umap, 1 row for pca
    n_cols = len(_color_by)
    fig, ax = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        constrained_layout=True,
        figsize=(panel_width * n_cols, panel_width * n_rows),
    )
    for j, proj_name in enumerate(['UMAP', 'PC']):
        if not use_umap:
            _ax = ax
            if proj_name == 'UMAP':
                continue
        else:
            _ax = ax[j]
        for i, col in enumerate(_color_by):            
                
                sns.scatterplot(
                    data=proj_df,
                    x=proj_name + '1',
                    y=proj_name + '2',
                    hue=col,
                    size=0.1,
                    ax=_ax[i],
                    alpha=1.0
                )

                _ax[i].set_title(f'{proj_name} - {col}')
                _ax[i].set_xlabel(proj_name + '1')
                _ax[i].set_ylabel(proj_name + '2')
                _ax[i].set_xticks([])
                _ax[i].set_yticks([])
                _ax[i].set_xticklabels([])
                _ax[i].set_yticklabels([])

                unique_values = proj_df[col].unique()
                if len(unique_values) > 10:
                    _ax[i].legend_.remove() # prevent legend from being too large

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_correlation_boxplot(
    corr_matrix,
    coorrelation_method='pearson',
    title="Reconstruction variability across resampled latent space",
    ylabel="Sample index",
    show_plot=True,
    save_path=None,
    figsize=(12, 10)
):
    """
    Plots a horizontal boxplot for correlation values between 
    ground truth and resampled gene expression values
    """
    plt.figure(figsize=figsize)
    sns.boxplot(data=corr_matrix, orient='h')
    xlabel = f"{coorrelation_method} correlation with ground truth"
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_perturb_cell_type(
    x_basis,
    meta_basis,
    x_reconst_perturb,
    meta_perturb,
    panel_width=5,
    from_cell_types=None,
    cell_type_col='cell_type',
    show_plot=True,
    save_path=None,
):
    pca = PCA(n_components=2)
    proj = pca.fit_transform(x_basis)
    proj_df = pd.DataFrame(
        proj[:,:2],
        columns=['PC1', 'PC2']
    )
    proj_df['cell_type'] = meta_basis[cell_type_col].values
    proj_df['cell_type_perturb_target'] = '-'

    all_cell_types = list(meta_basis[cell_type_col].unique())
    if 'random' in all_cell_types:
        all_cell_types.remove('random')

    if from_cell_types is None:
        from_cell_types = all_cell_types

    for _ct_source in from_cell_types:
        
        n_subplots = len(all_cell_types) - 1
        ncol = 4
        nrow = int(np.ceil(n_subplots / ncol))
        panel_width = 5

        fig, axs = plt.subplots(
            nrow,
            ncol,
            figsize=(ncol*panel_width, nrow*panel_width),
        )
        j = 0
        fig.suptitle(f'Perturb from {_ct_source}', fontsize=16)

        for _ct_target in all_cell_types:
            if _ct_source == _ct_target:
                continue

            _idx = np.intersect1d(
                np.where(meta_perturb['cell_type'] == _ct_source)[0],
                np.where(meta_perturb['cell_type_perturb_target'] == _ct_target)[0]
            )
            if len(_idx) == 0:
                continue

            _x_perturb = x_reconst_perturb[_idx,:]
            _meta_perturb = meta_perturb.iloc[_idx,:]

            perturb_proj = pca.transform(_x_perturb)
            perturb_proj_df = pd.DataFrame(
                perturb_proj[:,:2],
                columns=['PC1', 'PC2']
            )
            perturb_proj_df['cell_type'] = _meta_perturb['cell_type'].values
            perturb_proj_df['cell_type_perturb_target'] = _meta_perturb['cell_type_perturb_target'].values

            proj_df_concat = pd.concat(
                [proj_df, perturb_proj_df],
                axis=0
            )
            proj_df_concat = label_condition(
                proj_df_concat,
                _ct_source,
                _ct_target
            )

            ax = axs[j//ncol, j%ncol]
            sns.scatterplot(
                data=proj_df_concat,
                x='PC1',
                y='PC2',
                hue='perturbation',
                alpha=0.7,
                ax=ax,
                palette={
                    'source': 'blue',
                    'target': 'red',
                    'perturb': 'green',
                    '-': 'gray'
                },
            )
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_title(f'To {_ct_target}')
            j += 1
        
        if save_path is not None:
            plt.savefig(
                pathlib.Path(save_path) / f"{_ct_source}_perturbation.png", 
                dpi=300)
        if show_plot:
            plt.show()
        else:
            plt.close()