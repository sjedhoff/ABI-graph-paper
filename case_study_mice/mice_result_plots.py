################################################################################
##                                                                            ##
##                          Case Study Mice: Results                          ##
##                                                                            ##
################################################################################


import bayesflow as bf
import pyreadr
import numpy as np
import pandas as pd
import os

import bayesflow as bf
import keras
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path

def _load_first_df(rds_path: str):
    """Read an .rds file written by R and return the first (and usually only) DataFrame."""
    result = pyreadr.read_r(rds_path)
    return next(iter(result.values()))

def build_data_dict(
    file_path: str,
    data_attr: str,
    param_cols: list[str],
    splits: tuple[str, ...] = ("train", "val", "test"),
):
    """
    Build a dictionary like {'train_data': {...}, 'val_data': {...}, 'test_data': {...}}
    for a given simulation output.

    Parameters
    ----------
    file_path : str
        Directory containing the simulation_output_* files.
    data_attr : str
        Attribute suffix used in the file names, e.g. 'net_dens_ex_gr_20000'.
    param_cols : list[str]
        Column names to pull from the parameter DataFrame (e.g. ['network_density', 'growth_rate']).
    splits : tuple[str, ...], optional
        Which dataset splits to load.  Defaults to ("train", "val", "test").

    Returns
    -------
    dict
        Mapping split name + '_data' → dict with keys *param_cols* plus 'graph'.
    """
    data_dict = {}

    for split in splits:
        # ---- load graph array ----
        arr_path = os.path.join(file_path, f"simulation_output_array_{split}_{data_attr}.rds")
        array_xr = _load_first_df(arr_path)
        graphs = array_xr.values.transpose(2, 0, 1)  # (N, rows, cols)

        # ---- load parameters ----
        par_path = os.path.join(file_path, f"simulation_output_params_{split}_{data_attr}.rds")
        params_df = _load_first_df(par_path)

        # ---- assemble record ----
        record = {
            col: np.expand_dims(params_df[col].to_numpy(), axis=-1)
            for col in param_cols
        }
        record["graph"] = graphs
        data_dict[f"{split}_data"] = record

    return data_dict



### Day 5: Set Transformer
day = 5
run = 1

data_dict = build_data_dict(
    file_path=r"case_study_mice\data",
    data_attr=f"new_simulator_day_{day}_n50k_{run}",
    param_cols=["network_density", "exchange_factor"],
)

test_data_subset = {}
for k, v in data_dict["test_data"].items():
    a = np.asarray(v)
    if a.ndim == 0:
        test_data_subset[k] = a                         
    else:
        test_data_subset[k] = a[:100] 

filepath = Path("checkpoints/case_study_mice") / f"day_{day}_SetTransformer_{run}.keras"
approximator = keras.saving.load_model(filepath)

post_draws = approximator.sample(conditions=test_data_subset, num_samples=500)

var_names = [r"Network Density $\delta$", r"Exchange Factor $\alpha$"]

f = bf.diagnostics.recovery(estimates=post_draws, targets=test_data_subset, variable_names=var_names,
figsize=[7.4,4.8], uncertainty_agg_kwargs={"prob":0.8})
#f.savefig("plots/mice_settransformer_day_5_recovery.pdf")


post_draws_all =  approximator.sample(conditions=data_dict["test_data"], num_samples=500)

g = bf.diagnostics.calibration_ecdf(estimates=post_draws_all, targets=data_dict["test_data"], variable_names=var_names, difference=True,
figsize=[7.4,3.8], legend_fontsize=12)
for ax in g.axes:
    if legend := ax.get_legend():
        legend.remove()
#g.savefig("plots/mice_settransformer_day_5_calibration.pdf")


### Day 30: SetTransformer
day = 30
run = 1

data_dict_30 = build_data_dict(
    file_path=r"case_study_mice\data",
    data_attr=f"new_simulator_day_{day}_n50k_{run}",
    param_cols=["network_density", "exchange_factor"],
)

test_data_subset_30 = {}
for k, v in data_dict_30["test_data"].items():
    a = np.asarray(v)
    if a.ndim == 0:
        test_data_subset_30[k] = a                         
    else:
        test_data_subset_30[k] = a[:500] 

filepath = Path("checkpoints/case_study_mice") / f"day_{day}_SetTransformer_{run}.keras"
approximator_30 = keras.saving.load_model(filepath)

post_draws_30 = approximator_30.sample(conditions=test_data_subset_30, num_samples=500)

f = bf.diagnostics.recovery(estimates=post_draws_30, targets=test_data_subset_30, variable_names=var_names,
figsize=[7.4,4.8], legend_position=None)
#f.savefig("plots/mice_settransformer_day_30_recovery.pdf")

post_draws_30_all =  approximator_30.sample(conditions=data_dict_30["test_data"], num_samples=500)

g =  bf.diagnostics.calibration_ecdf(estimates=post_draws_30_all, targets=data_dict_30["test_data"], variable_names=var_names, difference=True,
figsize=[7.4,3.8], legend_location=None, legend_fontsize=None)
for ax in g.axes:
    if legend := ax.get_legend():
        legend.remove()
#g.savefig("plots/mice_settransformer_day_30_calibration.pdf")
