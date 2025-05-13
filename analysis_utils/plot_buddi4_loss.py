from pathlib import Path
from typing import Optional, Union

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_buddi4_loss(
        loss_df: pd.DataFrame, 
        show_plot: bool=True, 
        save_path: Optional[Union[str, Path]]=None,
        loss_columns: Optional[list]=None):
    
    _, axes = plt.subplots(1, 5, figsize=(25, 5), sharex=True)

    if loss_columns is None:
        
        loss_columns = [
            'X_reconstruction_loss', 
            'classifier_label_loss', 
            'classifier_stim_loss', 
            'classifier_samp_type_loss', 
            'prop_estimator_loss'
        ]

    for ax, col in zip(axes, loss_columns):
        sns.lineplot(
            data=loss_df, 
            x='index', 
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