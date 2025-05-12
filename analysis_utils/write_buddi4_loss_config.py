import yaml
from typing import Optional, Union, Any
from collections import defaultdict
from pathlib import Path

import pandas as pd    

def write_buddi4_loss_config(
    obj: Any,
    save_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    """
    
    loss_configs = defaultdict(list)
    
    for loss_name, loss_config in obj.losses.items():
        fn = type(loss_config[0]).__name__
        if fn == 'function':
            fn = loss_config[0].__name__        
        weight = loss_config[1]
        
        loss_configs['loss_name'].append(loss_name)
        loss_configs['loss_function'].append(fn)
        loss_configs['loss_weight'].append(weight)

    loss_configs = pd.DataFrame(
        loss_configs
    )

    if save_path is not None:
        
        if isinstance(save_path, str):
            save_path = Path(save_path)
        elif not isinstance(save_path, Path):
            raise ValueError("save_path must be a string or a Path object.")
        
        if save_path.suffix not in ['.yaml', '.yml']:
            raise ValueError("save_path must have a .yaml or .yml extension.")

        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            yaml.safe_dump(
                loss_configs.set_index('loss_name').to_dict(orient='index'), 
                f, sort_keys=False)
    
    return loss_configs