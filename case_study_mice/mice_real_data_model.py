################################################################################
##                                                                            ##
##                     Case Study Mice: Real World Data                       ##
##                                                                            ##
################################################################################

import pyreadr
import numpy as np
import pandas as pd
import os
import keras
import bayesflow as bf
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import random
from scipy.io import savemat

################################################################################
##                                  Training                                  ##
################################################################################
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


adapter = (
    bf.Adapter()
      .constrain("exchange_factor", lower=0.05, upper=0.95) 
      .rename("exchange_factor", "inference_variables") 
      .rename("graph",  "summary_variables")    
)



data_dict = build_data_dict(
    file_path=r"case_study_mice\data",
    data_attr=f"real_social_active_5_10000",
    param_cols=["exchange_factor"],
)


summary_net_st = bf.networks.SetTransformer()
inference_net_st = bf.networks.FlowMatching()

workflow_st = bf.BasicWorkflow(adapter=adapter,
                                inference_network=inference_net_st,
                                summary_network=summary_net_st)

early_stopping_st = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

history_st = workflow_st.fit_offline(data=data_dict["train_data"],
                                        validation_data=data_dict["val_data"],
                                        epochs=250, batch_size=128,
                                        callbacks=[early_stopping_st])



workflow_st.plot_default_diagnostics(test_data=data_dict["test_data"])

filepath = Path("checkpoints/case_study_mice") / f"real_social_active_5_10000.keras"
filepath.parent.mkdir(parents=True, exist_ok=True)
workflow_st.approximator.save(filepath=str(filepath))

################################################################################
##                           Performance diagnostics                          ##
################################################################################

# Load fitted model
file_approximator = Path("checkpoints/case_study_mice") / f"real_social_active_5_10000.keras"
approximator = keras.saving.load_model(file_approximator)


post_draws_test = approximator.sample(conditions=data_dict["test_data"], num_samples=500)

f = bf.diagnostics.recovery(estimates=post_draws_test, targets=data_dict["test_data"], 
                            variable_names="Recovery", title_fontsize=20, label_fontsize=18)
f.savefig("plots/mice_real_data_recovery.pdf")

f = bf.diagnostics.plots.calibration_ecdf(estimates=post_draws_test, targets=data_dict["test_data"],
                            variable_names="Calibration", title_fontsize=20, label_fontsize=18)
f.savefig("plots/mice_real_data_calibration.pdf")



################################################################################
##                                Real Datasets                               ##
################################################################################

# Load fitted model
file_approximator = Path("checkpoints/case_study_mice") / f"real_social_active_5_10000.keras"
approximator = keras.saving.load_model(file_approximator)


def read_data(filepath : str):
    graph = pyreadr.read_r(filepath)
    graph = next(iter(graph.values()))
    graph = graph.to_numpy()
    graph = np.expand_dims(graph, axis = 0)
    return {"graph" : graph}


social_graph = read_data(r"case_study_mice\real_data\data_ready\social_graph_selected.rds")

random.seed(31415)
post_draws =  approximator.sample(conditions=social_graph, num_samples=500)

# Plot posterior
draws = post_draws["exchange_factor"].squeeze()
kde = gaussian_kde(draws)
x = np.linspace(draws.min(), draws.max(), 300)

fig, ax = plt.subplots(figsize=(6.4, 4.8))
ax.plot(x, kde(x), color="#132a70", zorder=3)
ax.hist(draws, bins=30, density=True, color="#132a70", alpha=0.9, edgecolor="black", zorder=2)

ax.set_xlabel(r"Exchange Factor $\alpha$", fontsize=18)
ax.set_ylabel("Density", fontsize=18)
ax.set_title("Posterior Density", fontsize=20)

ax.grid(True, color="lightgrey", linewidth=0.5, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("plots/mice_real_data_posterior.pdf", bbox_inches="tight")
plt.show()

# Save the posterior draws for Posterior Predictive checks in R -> see case_study_mice/mice_real_data_PPC.R
savemat("case_study_mice/real_data/social_graph_posteriors.mat", post_draws)
