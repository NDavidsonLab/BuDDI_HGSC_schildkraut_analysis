from typing import Optional, List, Union, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from umap.umap_ import UMAP

def plot_perturb_reduction(
    x_basis: np.ndarray,
    meta_basis: pd.DataFrame,
    x_perturb: np.ndarray,
    meta_perturb: pd.DataFrame,
    down_sample_basis_n: Optional[int] = 500,
    filter_source: Optional[str] = None,
    filter_target: Optional[str] = None,
    color_by: Optional[List[str]] = None,
    reduction_method: str = 'PCA',
    reduction_kwargs: Optional[dict] = None,
    reduction_random_state: Optional[int] = None,
    panel_width: int = 5,
    alpha: float = 0.7,
    palette='tab20',
    title: Optional[str] = None,
    show_plot: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot reduction of perturbation data against basis data.
    This function performs dimensionality reduction on the basis and perturbation data,
    and creates a grid of scatter plots to visualize the results.

    :param x_basis: Basis data (2D array).
    :param meta_basis: Metadata for basis data (DataFrame).
    :param x_perturb: Perturbation data (2D array).
    :param meta_perturb: Metadata for perturbation data (DataFrame).
    :param down_sample_basis_n: Number of samples to downsample from basis data (int).
    :param filter_source: Filter for source cell type (str).
    :param filter_target: Filter for target cell type (str).
    :param color_by: List of columns to color by (list of str).
    :param reduction_method: Dimensionality reduction method ('PCA' or 'UMAP').
    :param reduction_kwargs: Additional arguments for the reduction method (dict).
    :param reduction_random_state: Random state for reproducibility (int).
    :param panel_width: Width of each panel (int).
    :param alpha: Alpha value for scatter plot (float).
    :param palette: Color palette for the plot (str).
    :param title: Title for the plot (str).
    :param show_plot: Whether to show the plot (bool).
    :param save_path: Path to save the plot (str).
    :return: None
    """
    # perturb type
    perturb_type = meta_perturb['perturb_type'].unique()
    if len(perturb_type) != 1:
        raise ValueError("meta_perturb['perturb_type'] must have exactly one unique value")
    perturb_type = perturb_type[0]

    if down_sample_basis_n is not None:
        counts = meta_basis[perturb_type].value_counts()
        min_count = counts.min()
        _meta_basis = meta_basis.copy()
        _meta_basis['_orig_idx'] = np.arange(len(_meta_basis))
        meta_basis = (
            _meta_basis
            .groupby(perturb_type, group_keys=False)
            .apply(lambda sub: sub.sample(n=min_count, random_state=42, replace=True))
        )
        idx = meta_basis['_orig_idx'].to_numpy()
        x_basis = x_basis[idx,:]
        meta_basis = meta_basis.drop(columns=['_orig_idx']).reset_index(drop=True)
    
    # 1) Filter perturb only
    mp = meta_perturb.copy()
    xp = x_perturb.copy()
    if filter_source is not None:
        mask = mp['source'] == filter_source
        xp, mp = xp[mask], mp[mask].reset_index(drop=True)
    if filter_target is not None:
        mask = mp['target'] == filter_target
        xp, mp = xp[mask], mp[mask].reset_index(drop=True)

    basis_source_idx = None
    if filter_source is not None:
        basis_source_idx = np.where(
            meta_basis[perturb_type] == filter_source
        )[0]

    # 2) Build color vars
    color_vars = ['target'] + (color_by or [])

    # 3) Validate presence
    for col in color_vars:
        if col not in meta_basis.columns or col not in mp.columns:
            raise ValueError(f"Column '{col}' must exist in both meta_basis and meta_perturb")

    # 4) PCA fit/transform
    if reduction_method == 'PCA':
        reducer = PCA(n_components=2, **(reduction_kwargs or {}))
        coords_basis = reducer.fit_transform(x_basis)
        coords_perturb = reducer.transform(xp)
    elif reduction_method == 'UMAP':
        pca_reducer = PCA(n_components=30)
        pca_basis = pca_reducer.fit_transform(x_basis)
        pca_perturb = pca_reducer.transform(xp)

        if reduction_random_state is not None:
            reduction_kwargs = reduction_kwargs or {}
            reduction_kwargs['random_state'] = reduction_random_state

        reducer = UMAP(n_components=2, **(reduction_kwargs or {}))
        coords_basis = reducer.fit_transform(pca_basis)
        coords_perturb = reducer.transform(pca_perturb)
    else:
        raise ValueError(f"Unsupported reduction method: {reduction_method}")

    # 5) Assemble DataFrames
    df_basis = pd.DataFrame(coords_basis, columns=[f'{reduction_method}1',f'{reduction_method}2'])
    df_basis = pd.concat([df_basis, meta_basis.reset_index(drop=True)], axis=1)
    df_basis['dataset'] = 'basis'
    df_basis['target'] = df_basis[perturb_type].astype(str)
    if basis_source_idx is not None:
        df_basis['identity'] = '-'
        df_basis.iloc[basis_source_idx, -1] = 'perturb source'

    df_perturb = pd.DataFrame(coords_perturb, columns=[f'{reduction_method}1',f'{reduction_method}2'])
    df_perturb = pd.concat([df_perturb, mp.reset_index(drop=True)], axis=1)
    df_perturb['dataset'] = 'perturb'

    # 6) Plot grid with shared axes
    n_rows = len(color_vars)
    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(2*panel_width, n_rows*panel_width),
        sharex=True, sharey=True,  # <— shared axes
        squeeze=False
    )

    for i, col in enumerate(color_vars):
        ax_ref, ax_only = axes[i]

        #  compute union of categories across both dataframes
        vals_basis  = df_basis[col].astype(str).unique().tolist()
        vals_pert   = df_perturb[col].astype(str).unique().tolist()
        levels      = list(dict.fromkeys(vals_basis + vals_pert))
        # build a palette of the right length
        colors      = sns.color_palette(palette, n_colors=len(levels))
        palette_dict = dict(zip(levels, colors))
            
        sns.scatterplot(
            data=df_basis,
            x=f'{reduction_method}1', y=f'{reduction_method}2',
            hue=col, 
            ax=ax_ref,
            legend='brief',
            alpha=alpha,
            hue_order=levels,
            palette=palette_dict,
            style='identity' if basis_source_idx is not None else None,
        )
        ax_ref.set_title(f"Reference (colored by '{col}')")

        # Perturb-only
        sns.scatterplot(
            data=df_perturb,
            x=f'{reduction_method}1', y=f'{reduction_method}2',
            hue=col,
            ax=ax_only,
            legend='brief',
            alpha=alpha,
            hue_order=levels,
            palette=palette_dict,
        )
        ax_only.set_title(f"Perturb only (colored by '{col}')")

    if title is not None:
        fig.suptitle(title, fontsize=16)
        fig.subplots_adjust(top=0.9)

    plt.tight_layout()

    # TODO: fix bottom caption wrapping and spacing
    # fig.subplots_adjust(bottom=0.15)
    # fig.text(
    #     0.5, 0.05,
    #     "Caption: This visualizes the dimensionality reduction of reconstructed gene expression"
    #     " after perturbation (right column) against that of reference basis data (left column)."
    #     " If most data points in the top right panel resemble the distribution and color pattern"
    #     " of the left panel (with the exception of top left pnael points marked as 'x'),"
    #     " the model training is likely successful and the perturbed latent space is sufficiently dis-entangled."
    #     ,
    #     ha="center", va="center",
    #     fontsize="small"
    # )

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close(fig)
    return None

def plot_corr_matrix_with_metadata(
    truth: Union[np.ndarray, pd.DataFrame],
    recon: Union[np.ndarray, pd.DataFrame],
    metadata: pd.DataFrame,
    meta_col: str,
    figsize: Tuple[float, float] = (10, 8),
    heatmap_cmap: str = 'vlag',
    cluster: bool = False,
    title: Optional[str] = None,
    show_plot: bool = True,
    save_path: Optional[str] = None
):
    """
    Compute and plot sample–sample correlation between ground truth and reconstructed data,
    **after sorting samples by `meta_col`** so that like-categories are contiguous.

    :param truth: Ground truth data (2D array or DataFrame).
    :param recon: Reconstructed data (2D array or DataFrame).
    :param metadata: Metadata DataFrame.
    :param meta_col: Column name in metadata to sort by.
    :param figsize: Size of the figure (tuple).
    :param heatmap_cmap: Colormap for the heatmap (str).
    :param cluster: Whether to cluster the heatmap (bool).
    :param title: Title for the plot (str).
    :param show_plot: Whether to show the plot (bool).
    :param save_path: Path to save the plot (str).
    :return: None
    """
    # ——— 1. Coerce to DataFrame ———
    # Align indices to metadata for consistency
    idx = metadata.index
    if isinstance(truth, np.ndarray):
        truth_df = pd.DataFrame(truth, index=idx)
    else:
        truth_df = truth.copy().loc[idx]
    if isinstance(recon, np.ndarray):
        recon_df = pd.DataFrame(recon, index=idx, columns=truth_df.columns)
    else:
        recon_df = recon.copy().loc[idx]

    # ——— 2. Sort all three by meta_col ———
    order = metadata[meta_col].sort_values().index
    truth_df = truth_df.loc[order]
    recon_df = recon_df.loc[order]
    meta_sorted = metadata.loc[order]

    # ——— 3. Standardize rows ———
    X = truth_df.values
    Y = recon_df.values
    Xn = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
    Yn = (Y - Y.mean(axis=1, keepdims=True)) / Y.std(axis=1, keepdims=True)

    # ——— 4. Compute correlation matrix ———
    corr = np.dot(Xn, Yn.T) / Xn.shape[1]
    corr_df = pd.DataFrame(corr, index=order, columns=order)

    # ——— 5. Build annotation colors based on sorted metadata ———
    meta = meta_sorted[meta_col]
    if pd.api.types.is_numeric_dtype(meta):
        norm = plt.Normalize(vmin=meta.min(), vmax=meta.max())
        cmap_meta = plt.cm.viridis
        colors = meta.map(lambda v: cmap_meta(norm(v)))
    else:
        categories = meta.unique()
        palette = sns.color_palette(n_colors=len(categories))
        lut = dict(zip(categories, palette))
        colors = meta.map(lut)

    # ——— 6. Plot ———
    g = sns.clustermap(
        corr_df,
        row_cluster=cluster,
        col_cluster=cluster,
        row_colors=colors,
        col_colors=colors,
        cmap=heatmap_cmap,
        figsize=figsize,
        vmin=-1, vmax=1, center=0,
    )

    # ——— 7. Legend / colorbar ———
    if pd.api.types.is_numeric_dtype(meta):
        sm = plt.cm.ScalarMappable(cmap=cmap_meta, norm=norm)
        sm.set_array([])
        g.figure.colorbar(
            sm, ax=g.ax_heatmap, orientation='vertical', label=meta_col
        )
    else:
        for cat, col in lut.items():
            g.ax_col_dendrogram.bar(
                0, 0, color=col, label=str(cat), linewidth=0
            )
        g.ax_col_dendrogram.legend(
            loc='center', ncol=1,
            bbox_to_anchor=(0.5, 1.1),
            title=meta_col
        )
    if title is not None:
        g.ax_heatmap.set_title(title, fontsize=16)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close(g.figure)
    return None 