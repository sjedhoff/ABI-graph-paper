################################################################################
##                                                                            ##
##                      Case Study Trains: Run Simulation                     ##
##                                                                            ##
################################################################################

import os
import numpy as np
import pyreadr
import bayesflow as bf
import matplotlib.pyplot as plt
import keras


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



file_path = "case_study_trains\\data"
data_attr = f"four_trains_sim_with_random_delay_large"
train_data, param_names = read_data(file_path, data_attr, n_trains = 4)

val_data, param_names = read_data(file_path, data_attr = f"{data_attr}_val", n_trains = 4)

adapter = (
    bf.Adapter()
      .concatenate(param_names, into="inference_variables")  
      .rename("graph", "summary_variables")               
)

n_epoch = 2

# ----------------- Run Model -----------------#
inf_network = bf.networks.CouplingFlow(transform="spline")


sum_network = bf.networks.SetTransformer(summary_dim=64)


workflow = bf.BasicWorkflow(
    adapter=adapter,
    inference_network=inf_network,
    summary_network=sum_network,
)


optimizer_1 = keras.optimizers.Adam()
workflow.approximator.compile(optimizer_1)
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)



history_1 = workflow.fit_offline(train_data, epochs=n_epoch, batch_size=1024, 
                validation_data=val_data, callbacks=[early_stopping])

fig = bf.diagnostics.loss(history_1, legend_fontsize=2, val_marker_size=1, label_fontsize=8)



# ------------------ Save Model  ------------------#
from pathlib import Path
filepath = Path("checkpoints/case_study_trains") / "train_sim_random_delay_10k_64each_earlystopping.keras"
filepath.parent.mkdir(exist_ok=True)
workflow.approximator.save(filepath=filepath)

approx = keras.saving.load_model(filepath)
