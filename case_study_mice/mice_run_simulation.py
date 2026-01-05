################################################################################
##                                                                            ##
##                   Case Study Mice: Run the simulations                     ##
##                                                                            ##
################################################################################

import pyreadr
import numpy as np
import pandas as pd
import os
import bayesflow as bf
import keras
from tensorflow.keras.callbacks import EarlyStopping
from bayesflow.utils.serialization import serializable
from pathlib import Path
import csv

### Load the data from the R simulations
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
      .constrain("network_density", lower=0.01, upper=0.5)
      .constrain("exchange_factor", lower=0.05, upper=0.5) 
      .concatenate(["network_density", "exchange_factor"], into="inference_variables") 
      .rename("graph",  "summary_variables")    
)


### Load the summary networks
from summary_networks.GCNInvariantLayer import GCNInvariantLayer
from summary_networks.GraphTransformerAttentionPooling import GraphTransformer


### Functions needed for saving results


CSV_PATH   = "all_metrics.csv"
ERROR_PATH = "errors.csv"

COLUMNS = [
    "day", "run", "workflow",
    "PC_ND","PC_EF","LG_ND","LG_EF","R_ND","R_EF","VAL_LOSS_LAST"
]

def ensure_header(path, columns):
    exists = os.path.exists(path)
    if not exists:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            f.flush(); os.fsync(f.fileno())
    return exists

def load_existing_keys(path, key_cols=("dataset","day","run","workflow")):
    """Build a set of keys already present, so we can skip duplicates on resume."""
    keys = set()
    if not os.path.exists(path):
        return keys
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            keys.add(tuple(row[k] for k in key_cols))
    return keys

def append_row_atomic(path, row_dict, columns=COLUMNS):
    """Append a single row and fsync so progress is never lost."""
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writerow(row_dict)
        f.flush(); os.fsync(f.fileno())

def log_error(error_path, meta, exc):
    os.makedirs(os.path.dirname(error_path) or ".", exist_ok=True)
    err_row = {**meta, "error": str(exc)}
    header  = list(meta.keys()) + ["error"]
    if not os.path.exists(error_path):
        with open(error_path, "w", newline="", encoding="utf-8") as ef:
            w = csv.DictWriter(ef, fieldnames=header); w.writeheader()
            ef.flush(); os.fsync(ef.fileno())
    with open(error_path, "a", newline="", encoding="utf-8") as ef:
        w = csv.DictWriter(ef, fieldnames=header); w.writerow(err_row)
        ef.flush(); os.fsync(ef.fileno())



def compute_metrics_for_workflow(ap, test_data):
    """
    Return dict with keys:
    PC_ND, PC_EF, LG_ND, LG_EF, R_ND, R_EF, VAL_LOSS_LAST.
    This mirrors the code you already have.
    """
    # Sample
    est = ap.sample(num_samples=500, conditions=test_data)

    # Get estimates/targets w/o leaving plots open
    pd = bf.utils.plot_utils.prepare_plot_data(
        estimates=est, targets=test_data,
        variable_keys=None, variable_names=None, num_col=None, num_row=None, figsize=None,
    )
    estimates = np.asarray(pd["estimates"])
    targets   = np.asarray(pd["targets"])

    # Point estimate: median over sample axis (axis with size 500 if present)
    num_samples = 500
    sample_axis = list(estimates.shape).index(num_samples) if num_samples in estimates.shape else 1
    point_est = np.median(estimates, axis=sample_axis)
    if point_est.shape[0] != targets.shape[0] and point_est.shape[1] == targets.shape[0]:
        point_est = point_est.T

    # Correlations (first two params)
    n_params = min(targets.shape[1], point_est.shape[1], 2)
    r_vals = [np.corrcoef(targets[:, j], point_est[:, j])[0, 1] for j in range(n_params)]
    while len(r_vals) < 2:
        r_vals.append(np.nan)

    # Diagnostics
    pc = bf.diagnostics.posterior_contraction(estimates=est, targets=test_data)
    lg = bf.diagnostics.metrics.calibration_log_gamma(estimates=est, targets=test_data)
    pc_vals = np.asarray(pc.get("values", pc)).ravel()
    lg_vals = np.asarray(lg.get("values", lg)).ravel()

    # Validation loss now (if you can evaluate); else np.nan
    def eval_val_loss_now(ap):
        for attr in ("model","amortizer","_amortizer","inference_net","network"):
            if hasattr(ap, attr):
                obj = getattr(ap, attr)
                model = getattr(obj, "model", obj)
                if hasattr(model, "evaluate") and "val_inputs" in globals() and "val_targets" in globals():
                    return float(model.evaluate(val_inputs, val_targets, verbose=0))
        return np.nan

    val_last = eval_val_loss_now(ap)

    return {
        "PC_ND": float(pc_vals[0]) if pc_vals.size > 0 else np.nan,
        "PC_EF": float(pc_vals[1]) if pc_vals.size > 1 else np.nan,
        "LG_ND": float(lg_vals[0]) if lg_vals.size > 0 else np.nan,
        "LG_EF": float(lg_vals[1]) if lg_vals.size > 1 else np.nan,
        "R_ND":  float(r_vals[0]),
        "R_EF":  float(r_vals[1]),
        "VAL_LOSS_LAST": float(val_last),
    }

VAL_CSV = "val_losses.csv"

def append_val_row(day, run, last_vals, path=VAL_CSV):
    
    """
    last_vals: dict with keys ['gcn','ds','st','gt'] -> floats
    Appends one row; writes header if file doesn't exist.
    """
    row = np.array([[day, run,
                     last_vals["gcn"], last_vals["ds"],
                     last_vals["st"],  last_vals["gt"]]], dtype=float)
    header = "day,run,gcn,ds,st,gt"
    file_exists = os.path.exists(path)
    with open(path, "a") as f:
        np.savetxt(f, row, delimiter=",",
                   header=(header if not file_exists else ""), comments="")


### Hyperparameters
days = [5, 10, 30]
runs = [1, 2, 3, 4, 5]

n_epoch = 2

for day in days:
    for run in runs:
        # --- load the data ---
        data_dict = build_data_dict(
            file_path=r"case_study_mice\data",
            data_attr=f"new_simulator_day_{day}_n50k_{run}",
            param_cols=["network_density", "exchange_factor"],
        )

        # --- build workflows ---
        # GCN
        summary_net_gcn = GCNInvariantLayer()
        inference_net_gcn = bf.networks.CouplingFlow(transform="spline")
        workflow_gcn = bf.BasicWorkflow(adapter=adapter,
                                        inference_network=inference_net_gcn,
                                        summary_network=summary_net_gcn)
        early_stopping_gcn = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
        history_gcn = workflow_gcn.fit_offline(data=data_dict["train_data"],
                                               validation_data=data_dict["val_data"],
                                               epochs=n_epoch, batch_size=128,
                                               callbacks=[early_stopping_gcn])

        # Deep Sets
        summary_net_ds = bf.networks.DeepSet()
        inference_net_ds = bf.networks.CouplingFlow(transform="spline")
        workflow_ds = bf.BasicWorkflow(adapter=adapter,
                                       inference_network=inference_net_ds,
                                       summary_network=summary_net_ds)
        early_stopping_ds = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
        history_ds = workflow_ds.fit_offline(data=data_dict["train_data"],
                                             validation_data=data_dict["val_data"],
                                             epochs=n_epoch, batch_size=128,
                                             callbacks=[early_stopping_ds])

        # Set Transformer
        summary_net_st = bf.networks.SetTransformer()
        inference_net_st = bf.networks.CouplingFlow(transform="spline")
        workflow_st = bf.BasicWorkflow(adapter=adapter,
                                       inference_network=inference_net_st,
                                       summary_network=summary_net_st)
        early_stopping_st = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
        history_st = workflow_st.fit_offline(data=data_dict["train_data"],
                                             validation_data=data_dict["val_data"],
                                             epochs=n_epoch, batch_size=128,
                                             callbacks=[early_stopping_st])

        # Graph Transformer
        summary_net_gt = GraphTransformer()
        inference_net_gt = bf.networks.CouplingFlow(transform="spline")
        workflow_gt = bf.BasicWorkflow(adapter=adapter,
                                       inference_network=inference_net_gt,
                                       summary_network=summary_net_gt)
        early_stopping_gt = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
        history_gt = workflow_gt.fit_offline(data=data_dict["train_data"],
                                             validation_data=data_dict["val_data"],
                                             epochs=n_epoch, batch_size=128,
                                             callbacks=[early_stopping_gt])

        # --- save approximators ---
        wf_names  = ["GCN", "DeepSet", "SetTransformer", "GraphTransformer"]
        wf_names  = ["SetTransformer"]
        
        for i in range(4):
            filepath = Path("checkpoints/case_study_mice") / f"day_{day}_{wf_names[i]}_{run}.keras"
            filepath.parent.mkdir(parents=True, exist_ok=True)
            workflows[i].approximator.save(filepath=str(filepath))  # cast to str

        # --- save LAST validation loss for each workflow (append safely) ---
        histories = [history_gcn, history_ds, history_st, history_gt]
        last_vals = {}
        for name, h in zip(["gcn","ds","st","gt"], histories):
            vl = h.history.get("val_loss", [])
            last_vals[name] = float(vl[-1]) if len(vl) else np.nan
        append_val_row(day, run, last_vals, path=VAL_CSV)

        # --- compute & save metrics  ---
        # Use test/holdout data; adjust if you prefer val_data
        test_data = data_dict.get("test_data", data_dict["val_data"])
        for wf_name, wf in zip(wf_names, workflows):
            key = (str(day), str(run), str(wf_name))
            meta = {"day": day, "run": run, "workflow": wf_name}
            try:
                # if your function expects an approximator, pass wf.approximator
                m = compute_metrics_for_workflow(wf.approximator, test_data)
                row = {**meta, **m}
                append_row_atomic(CSV_PATH, row)
            except Exception as e:
                log_error(ERROR_PATH, meta, e)
                continue
