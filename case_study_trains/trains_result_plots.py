################################################################################
##                                                                            ##
##                        Case Study Trains: Results                          ##
##                                                                            ##
################################################################################

import keras
import bayesflow as bf
import numpy as np
import matplotlib.pyplot as plt
import os
import pyreadr
from scipy.stats import gaussian_kde 

#---------------------- Loading data and approximator --------------------------#
from pathlib import Path
filepath = Path("checkpoints/case_study_trains") / "train_sim_random_delay_10k_64each_earlystopping.keras"
approx_model = keras.saving.load_model(filepath)

def _load_first_df(rds_path: str):
    result = pyreadr.read_r(rds_path)
    return next(iter(result.values()))

def read_data(file_path, data_attr, n_trains):
    # ---- graphs: keep as (N, 10, 27) ----
    arr_path = os.path.join(file_path, f"graph_{data_attr}.rds")
    array_xr = _load_first_df(arr_path)                 # pandas DataFrame / xarray-like
    graphs = array_xr.values.transpose(2, 0, 1)         # -> (N, 10, 27)
    graphs = graphs.astype("float32")

    # ---- parameters: 4 rows x N cols -> dict of (N,1) arrays ----
    par_path = os.path.join(file_path, f"inference_vars_{data_attr}.rds")
    params_df = _load_first_df(par_path)                # shape (n_trains, N)
    P = params_df.to_numpy()                            # (n_trains, N)


    names = [f"total_time_{i}" for i in range(1, n_trains+1)]
    data = {names[i]: P[i].astype("float32")[:, None] for i in range(n_trains)}  # each (N,1)

    # add graphs unchanged
    data["graph"] = graphs
    return data, names


data_attr = f"four_trains_sim_with_random_delay_large"
test_data, param_names = read_data(file_path="case_study_trains\\data",
                                     data_attr = f"{data_attr}_test", n_trains=4)

estimates = approx_model.sample(conditions=test_data, num_samples=500)

# make a subset of the training data
test_data_subset = {}
for k, v in test_data.items():
    a = np.asarray(v)
    if a.ndim == 0:
        test_data_subset[k] = a                         
    else:
        test_data_subset[k] = a[:100] 


test_data_post_subset = approx_model.sample(num_samples=500, conditions=test_data_subset)

var_names = variable_names=["Train 1", "Train 2", "Train 3", "Train 4"]

#---------------------- Plots for the paper --------------------------#

f = bf.diagnostics.plots.recovery(estimates=test_data_post_subset, targets=test_data_subset, figsize=[14.8,3.8],
            variable_names=var_names)
#f.savefig("plots/train_simulator_paper_recovery.pdf")

p = bf.diagnostics.plots.calibration_ecdf(estimates=estimates, targets=test_data, figsize=[14.4,3.6],
            variable_names=["", "", "", ""], difference=True, legend_fontsize=11)

#p.savefig("plots/train_simulator_paper_calibration.pdf")

def plot_posterior_vs_ground_truth_density(
    schedule_id,
    data_attr_gt,
    workflow,
    test_data,
    legend = True,
    header = True,
    figsize=(14, 3.8),
    label_fontsize: int = 16,
    title_fontsize: int = 18,
    metric_fontsize: int = 16,  # used for legend here
    tick_fontsize: int = 12,
    gt_color: str = '#02b8b8',   # main color
    post_color: str = '#132a70'  # secondary color
):
    par_path = os.path.join(file_path, f"ground_truth_{data_attr_gt}.rds")
    params_df = _load_first_df(par_path)
    ground_truth = params_df.to_numpy()    # (N, n_sample, 4)

    post_draws = workflow.sample(num_samples=500, conditions=test_data)
    post_array = np.concatenate(list(post_draws.values()), axis=-1)  # (N, n_sample, D)

    # 1 x 4 layout, controlled by figsize
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=figsize)
    axes = np.atleast_1d(axes)

    x_min = np.min([ground_truth[schedule_id].min(), post_array[schedule_id].min()])
    x_max = np.max([ground_truth[schedule_id].max(), post_array[schedule_id].max()])
    x_grid = np.linspace(x_min-10, x_max+10, 200)

    for i, ax in enumerate(axes):
        gt_vals = ground_truth[schedule_id, :, i]
        post_vals = post_array[schedule_id, :, i]

        gt_kde = gaussian_kde(gt_vals)
        post_kde = gaussian_kde(post_vals)

        # colors + line styles
        ax.plot(x_grid, gt_kde(x_grid), label='Ground Truth',
                alpha=0.9, linewidth=2, color=gt_color)
        ax.plot(x_grid, post_kde(x_grid), label='Estimated Posterior',
                alpha=0.9, linewidth=2, linestyle='--', color=post_color)

        if header:
            ax.set_title(f'Train {i+1}', fontsize=title_fontsize)
        ax.set_xlabel('Total travel time', fontsize=label_fontsize)
        if i == 0:
            ax.set_ylabel('Density', fontsize=label_fontsize)
            #leg = ax.legend(fontsize=metric_fontsize)
        else:
            ax.set_ylabel('')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # tick font sizes
        ax.tick_params(axis='both', labelsize=tick_fontsize)

    
    #plt.tight_layout(rect=[0, 0.22, 1, 1])   # 22% of the figure reserved at bottom

    # 2) Get legend entries from one axis
    handles, labels = axes[0].get_legend_handles_labels()

    #fig.subplots_adjust(bottom=0.28)

    # 3) Put a single horizontal legend *below* the x-axis labels
    if legend:
        fig.legend(
            handles,
            labels,
            loc='lower center',          # center at bottom
            bbox_to_anchor=(0.5, -0.15),  
            ncol=2,
            fontsize=metric_fontsize,
            frameon=True,
        )


    return fig, axes

file_path="case_study_trains\\data"
fig, axes = plot_posterior_vs_ground_truth_density(schedule_id=3, data_attr_gt=f"{data_attr}_test",
                        workflow=approx_model, test_data=test_data, legend=True, header=False)

fig.savefig("plots/train_simulator_paper_posteriors_vs_groundtruth.pdf")


fig, axes = plot_posterior_vs_ground_truth_density(schedule_id=0, data_attr_gt=f"{data_attr}_test",
                        workflow=approx_model, test_data=test_data, legend=False, figsize=(14, 3.2),)

fig.savefig("plots/train_simulator_paper_posteriors_vs_groundtruth_2.pdf")