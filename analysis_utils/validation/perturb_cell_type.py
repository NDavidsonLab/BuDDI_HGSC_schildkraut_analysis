from typing import Iterable, Optional, List, Any
import warnings

from .perturb_y import _perturb_y

def perturb_single_cell_type(
    obj: Any,
    data: Any,
    n_subsamples: Optional[int] = 500,
    n_resamples: int = 1,
    ignore_cell_types: Iterable[str] = [],
    integrate_over: Optional[List[str]] = None,
    seed: Optional[int] = 42,
    query_kwargs: Optional[dict] = None,
):
    
    """
    Perform buddi4 model validation with cell type proportion perturbation experiment, 
    by swapping proportion matrices of pseudobulk entries corresponding to different cell types.

    :param obj: BuDDI4 model object.
    :param data: BuDDI4Data object containing the data.
    :param n_subsamples: Number of subsamples to draw from each source cell type. 
        Should be relatively small (500-1000) to avoid memory issues.
        Passed on to the _perturb_y function instead of here in the data.get() due to needing to
        sub-sample per cell type as opposed to the entire dataset.
        If None, _perturb_y will try to use all samples but will resort to down-sampling to get
        source and target perturbation cell types to be the same size.
    :param n_resamples: Number of resamples to draw from each source cell type.
        Should be 1 as re-sampling in single cell pseudo-bulk is largely unnecessary.
    :param integrate_over: List of latent spaces to integrate over.
        If None, no integration is performed.
    :param ignore_cell_types: List of cell types to ignore during perturbation.
    :param seed: Random seed for reproducibility.
    :return: Perturbed cell type proportions.
    """
    
    query_kwargs = query_kwargs or {}
    if 'samp_type' in query_kwargs:
        if query_kwargs['samp_type'] != 'sc_ref':
            warnings.warn("Overriding query_kwargs['samp_type'] to 'sc_ref'")
            query_kwargs['samp_type'] = 'sc_ref'
    else:
        query_kwargs['samp_type'] = 'sc_ref'

    try:
        data_query = data.query(
            **query_kwargs
        )
    except Exception as e:
        raise ValueError(f"Error querying data: {e}")
    
    try:
        (X, y, meta) = data_query.get(
            ('X', 'Y', 'meta')
        )
    except Exception as e:
        raise ValueError(f"Error retrieving data from the BuDDI4Data object: {e}")
    
    try:
        cell_type_names = data.cell_type_names
    except AttributeError:
        raise ValueError("Data object must have 'cell_type_names' attribute.")
    if not all(ct in cell_type_names for ct in ignore_cell_types):
        warnings.warn(
            f"Some cell types in 'ignore_cell_types' not found in 'cell_type_names': {set(ignore_cell_types) - set(cell_type_names)}"
        )

    cell_type_col = data.ct_column

    return _perturb_y(
        obj=obj,
        X=X,
        y=y,
        meta=meta,
        cell_type_names=cell_type_names,
        cell_type_col=cell_type_col,
        n_subsamples=n_subsamples,
        n_resamples=n_resamples,
        integrate_over=integrate_over,
        ignore_cell_types=ignore_cell_types,
        seed=seed,
    )