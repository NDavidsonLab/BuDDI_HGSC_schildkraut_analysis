import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

def _get_projection(z, type='PCA'):
    if type == 'PCA':
        proj = PCA(n_components=2)
    elif type == 'UMAP':
        proj = umap.UMAP(n_components=2)
    elif type == 'tSNE':
        proj = TSNE(n_components=2)
    else:
        raise ValueError(f"Unknown projection type {type}")
    
    proj_df = pd.DataFrame(
        proj.fit_transform(z),
        columns=[f'{type}_0', f'{type}_1']
    )

    return proj_df

def _plot_projection_marker(
        proj_df, 
        color_vec, 
        ax, 
        title="", 
        alpha=0.3, 
        legend_title="", 
        marker_vec=None,
        palette="hls"
    ):
    proj_df[legend_title] = color_vec

    if marker_vec is not None:
        proj_df['marker'] = marker_vec

    g = sns.scatterplot(
        x=proj_df.columns[0], y=proj_df.columns[1],
        data=proj_df,
        hue=legend_title,
        style='marker' if marker_vec is not None else None,
        palette=sns.color_palette(palette, len(np.unique(color_vec))),
        legend="full",
        alpha=alpha, ax= ax
    )

    ax.set_title(title)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    return g

def plot_buddi4_latent_space(
    model,
    train_data,
    type='PCA',
    n_subsample=500,
    replace_subsample=True,
    figsize=None,
    panel_width=5,
    show_plot=True,
    save_path=None,
    palette="hls",
    alpha=1.0
):
    
    # Subsample the data to reduce compute resource usage
    kp_idx = np.random.choice(
        range(train_data.X_kp.shape[0]), 
        n_subsample, 
        replace=replace_subsample)
    unkp_idx = np.random.choice(
        range(train_data.X_unkp.shape[0]), 
        n_subsample, 
        replace=replace_subsample)
    
    _meta = pd.concat([
        train_data.meta_kp.iloc[kp_idx],
        train_data.meta_unkp.iloc[unkp_idx]
        ],
        axis=0
    )
    
    pred = model(
        np.concatenate(
            [train_data.X_kp[kp_idx], train_data.X_unkp[unkp_idx]],
            axis=0
        )
    )
    # extract the latent spaces
    _, z_label, z_stim, z_samp_type, z_slack, _, _, _, y_hat = pred
    latent_spaces = [z_label, z_stim, z_samp_type, z_slack]
    latent_spaces = [
        latent_space[:,:latent_space.shape[1]//2] # trim away the logvar
        for latent_space in latent_spaces
    ]
    latent_spaces = [y_hat] + latent_spaces
    latent_space_names = [
        'Cell Type', 
        'Sample ID', 
        'Perturbation', 
        'Technology', 
        'Slack']

    # Define the color vectors for scatter plot hue
    label_vec = _meta['sample_id'].values
    stim_vec = _meta['stim'].values
    samp_type_vec = _meta['samp_type'].values

    # for cell type:
    if 'cell_type' in _meta.columns:
        # if detailed cell type information is available, use that
        cell_prop_vec = _meta['cell_type'].values
        marker_vec = _meta['cell_prop_type'].values
    else:
        # otherwise, use the general cell type information 
        #(single cell dominant vs random)
        cell_prop_vec = _meta['cell_prop_type'].values
        marker_vec = None

    color_vecs = [cell_prop_vec, label_vec, stim_vec, samp_type_vec]
    color_legend_names = latent_space_names[:-1] # do not include slack

    if figsize is None:
        figsize = (
            panel_width*len(latent_spaces), 
            panel_width*len(color_vecs))
    
    fig, axs = plt.subplots(
        len(color_vecs), 
        len(latent_spaces), 
        figsize=figsize)

    for i, (latent_space_name, latent_space) in enumerate(
        zip(latent_space_names, latent_spaces)):

        proj_df = _get_projection(latent_space, type=type)

        for j, (color_legend_name, color_vec) in enumerate(zip(color_legend_names, color_vecs)):

            _plot_projection_marker(
                proj_df=proj_df,
                color_vec=color_vec,
                marker_vec=marker_vec,
                ax=axs[j, i],
                title=latent_space_name if j == 0 else "", 
                alpha=alpha, 
                legend_title=color_legend_name,
                palette=palette
            )

            # Remove legend for all but the last column in each row
            if i != len(latent_spaces) - 1 or color_legend_name == 'Sample ID': # do not display legend for the sample ids
                axs[j, i].get_legend().remove()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig

def plot_train_loss(
        loss_df, 
        loss_columns=None,
        show_plot=True, 
        save_path=None,
        panel_width=5,
        ):

    if loss_columns is None:
        loss_columns = [
            'X_reconstruction_loss', 
            'classifier_label_loss', 
            'classifier_stim_loss', 
            'classifier_samp_type_loss', 
            'prop_estimator_loss'
        ]
    
    if '_index' not in loss_df.columns:
        loss_df['_index'] = loss_df['epoch'] * loss_df['batch'].max() + loss_df['batch']

    fig, axes = plt.subplots(
        1, len(loss_columns), 
        figsize=(panel_width * len(loss_columns), panel_width), sharex=True)

    for ax, col in zip(axes, loss_columns):
        sns.lineplot(
            data=loss_df, 
            x='_index', 
            y=col, 
            hue='type', 
            ax=ax
        )
        ax.set_title(col)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close()