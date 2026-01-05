################################################################################
##                                                                            ##
##                       Case Study Toy: Run Simulation                       ##
##                                                                            ##
################################################################################

import pyreadr
import numpy as np
import pandas as pd
import os
import keras
import bayesflow as bf
from tensorflow.keras.callbacks import EarlyStopping
from bayesflow.utils.serialization import serializable
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import csv
import random


### ---- Simulator and Priors --- ###

N_NODES = 30

def prior():
    pi_aa = np.random.uniform(low=0.1, high=0.9)
    pi_bb = np.random.uniform(low=0.1, high=0.9)
    pi_ab = np.random.uniform(low=0.1, high=0.9)
    gamma = np.random.uniform(low=0.1, high=0.9)

    num_a = np.random.randint(low=5, high=25)
    return dict(pi_aa=pi_aa, pi_bb=pi_bb, pi_ab=pi_ab, gamma=gamma, num_a=num_a)



def simulator(pi_aa, pi_bb, pi_ab, gamma, num_a, N_NODES = N_NODES):

    adj = np.zeros((N_NODES, N_NODES), dtype=np.int8)
    common = np.zeros((N_NODES, N_NODES), dtype=np.int8)
    x = np.zeros((N_NODES, 1), dtype=np.int8)
    x[:num_a, 0] = 1

    # base probabilities
    for i in range(N_NODES - 1):
        for j in range(i + 1, N_NODES):
            if x[i, 0] == x[j, 0]:
                p_edge = pi_aa if x[i, 0] == 1 else pi_bb
            else:
                p_edge = pi_ab

            adj[i, j] = np.random.binomial(1, p_edge)
            adj[j, i] = adj[i, j]

    # count common neighbors
    for i in range(N_NODES - 1):
        for j in range(i + 1, N_NODES):
            common[i, j] = np.sum(adj[i, :] * adj[j, :])
    
    # triadic closure
    for i in range(N_NODES - 1):
        for j in range(i + 1, N_NODES):
            if (adj[i, j] == 0) and (common[i, j] > 0):
                adj[i, j] = np.random.binomial(1, gamma)
                adj[j, i] = adj[i, j]

    graph_tensor = np.concatenate([x, adj.astype(np.float32)], axis=-1)
    return {"graph": graph_tensor, "common" : common}



sim = bf.make_simulator([prior, simulator])

adapter = (
    bf.Adapter()
      .constrain("pi_aa", lower=0.1, upper=0.9)
      .constrain("pi_bb", lower=0.1, upper=0.9)
      .constrain("pi_ab", lower=0.1, upper=0.9)
      .constrain("gamma", lower=0.1, upper=0.9)

      .concatenate(["pi_aa", "pi_bb", "pi_ab", "gamma"], into="inference_variables")  
      .rename("graph", "summary_variables")  
)

### --- Load all different Summary networks --- ###
from summary_networks.DeepSetMeanPooling import DeepSetMeanPooling
from summary_networks.DeepSetAttentionPooling import DeepSetMHA

from summary_networks.GCNMeanPooling import GCNMeanPooling
from summary_networks.GCNInvariantLayer import GCNInvariantLayer
from summary_networks.GCNAttentionPooling import GCNMHA

from summary_networks.GraphTransformerMeanPooling import GraphTransformerMeanPooling
from summary_networks.GraphTransformerInvariantLayer import GraphTransformerInvariantLayer
from summary_networks.GraphTransformerAttentionPooling import GraphTransformer

from summary_networks.SetTransformerMeanPooling import SetTransformerMeanPooling
from summary_networks.SetTransformerInvariantLayer import SetTransformerInvariantLayer



### --- hyperparameter --- ###
runs = [1, 2, 3, 4, 5]

epochs = 250

random.seed(31415)
test_data = sim.sample(500)


### --- functions for saving the results --- ###

CSV_PATH   = "case_study_toy/results/all_metrics.csv"
ERROR_PATH = "case_study_toy/results/errors.csv"
VAL_CSV    = "case_study_toy/results/val_losses.csv"

COLUMNS = [
    "run", "workflow",
    "PC_pi_aa","PC_pi_bb", "PC_pi_ab", "PC_gamma",
    "LG_pi_aa","LG_pi_bb", "LG_pi_ab", "LG_gamma",
    "R_pi_aa", "R_pi_bb", "R_pi_ab", "R_gamma",
    "number_parameters",
]

def ensure_header(path, columns):
    exists = os.path.exists(path)
    if not exists:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            f.flush(); os.fsync(f.fileno())
    return exists


def append_val_row(workflow_name, run, last_loss, path=VAL_CSV):
    """
    Append eine Zeile in val_losses.csv, header-sicher und atomar.
    """
    ensure_val_header(path)
    row = {"workflow": workflow_name, "run": run, "last_loss": float(last_loss) if last_loss is not None else np.nan}
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["workflow","run","last_loss"])
        w.writerow(row)
        f.flush(); os.fsync(f.fileno())

def ensure_val_header(path=VAL_CSV):
    header = ["workflow", "run", "last_loss"]
    ensure_header(path, header)

def load_existing_keys(path, key_cols=("run","workflow")):
    """Build a set of keys already present, so we can skip duplicates on resume."""
    keys = set()
    if not os.path.exists(path):
        return keys
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            keys.add(tuple(row.get(k, "") for k in key_cols))
    return keys

def append_row_atomic(path, row_dict, columns=COLUMNS):
    """Append a single row and fsync so progress is never lost."""
    # Sicherstellen, dass der Header existiert
    ensure_header(path, columns)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writerow(row_dict)
        f.flush(); os.fsync(f.fileno())

def log_error(error_path, meta, exc):
    os.makedirs(os.path.dirname(error_path) or ".", exist_ok=True)
    err_row = {**meta, "error": str(exc)}
    header  = list(meta.keys()) + ["error"]
    new_file = not os.path.exists(error_path)
    with open(error_path, "a", newline="", encoding="utf-8") as ef:
        w = csv.DictWriter(ef, fieldnames=header)
        if new_file:
            w.writeheader()
        w.writerow(err_row)
        ef.flush(); os.fsync(ef.fileno())




def compute_metrics_for_workflow(ap, test_data):
    """
    Return dict with keys:
    PC_* , LG_* , R_*  (siehe COLUMNS)
    """
    # Sample
    est = ap.sample(num_samples=500, conditions=test_data)

    # Estimates/Targets vorbereiten
    pd = bf.utils.plot_utils.prepare_plot_data(
        estimates=est, targets=test_data,
        variable_keys=None, variable_names=None, num_col=None, num_row=None, figsize=None,
    )
    estimates = np.asarray(pd["estimates"])
    targets   = np.asarray(pd["targets"])

    # Punkt-Schätzer: Median über Sample-Achse
    num_samples = 500
    point_est = np.median(estimates, axis=1)

    # Shapes angleichen (falls transponiert)
    if point_est.shape[0] != targets.shape[0] and point_est.shape[1] == targets.shape[0]:
        point_est = point_est.T

    # Korrelationen für bis zu 4 Parameter
    n_params = min(targets.shape[1], point_est.shape[1], 4)
    r_vals = []
    for j in range(n_params):
        # robust gegen konstante Spalten
        t = targets[:, j]
        p = point_est[:, j]
        if np.std(t) == 0 or np.std(p) == 0:
            r_vals.append(np.nan)
        else:
            r_vals.append(float(np.corrcoef(t, p)[0, 1]))
    # auf 4 auffüllen
    while len(r_vals) < 4:
        r_vals.append(np.nan)

    # Diagnostics
    pc = bf.diagnostics.posterior_contraction(estimates=est, targets=test_data)
    lg = bf.diagnostics.metrics.calibration_log_gamma(estimates=est, targets=test_data)
    pc_vals = np.asarray(pc.get("values", pc)).ravel()
    lg_vals = np.asarray(lg.get("values", lg)).ravel()


    # num of parameters
    num_par = ap.count_params()

    return {
        "number_parameters" : num_par,
        "PC_pi_aa": float(pc_vals[0]) if pc_vals.size > 0 else np.nan,
        "PC_pi_bb": float(pc_vals[1]) if pc_vals.size > 1 else np.nan,
        "PC_pi_ab": float(pc_vals[2]) if pc_vals.size > 2 else np.nan,
        "PC_gamma": float(pc_vals[3]) if pc_vals.size > 3 else np.nan,
        "LG_pi_aa": float(lg_vals[0]) if lg_vals.size > 0 else np.nan,
        "LG_pi_bb": float(lg_vals[1]) if lg_vals.size > 1 else np.nan,
        "LG_pi_ab": float(lg_vals[2]) if lg_vals.size > 2 else np.nan,
        "LG_gamma": float(lg_vals[3]) if lg_vals.size > 3 else np.nan,
        "R_pi_aa":  r_vals[0],
        "R_pi_bb":  r_vals[1],
        "R_pi_ab":  r_vals[2],
        "R_gamma":  r_vals[3],
    }

### --- Running the simulation --- ###

for run in runs:
    ## Define all workflows:

    #-- Deep Sets--#
    sum_net_ds_mp = DeepSetMeanPooling(summary_dim=16, mlp_widths_equivariant=(64,64))
    inf_net_ds_mp = bf.networks.CouplingFlow(transform="spline")

    workflow_ds_mp = bf.BasicWorkflow(
        simulator=sim,
        adapter=adapter,
        summary_network=sum_net_ds_mp,
        inference_conditions=inf_net_ds_mp
    )


    sum_net_ds_il = bf.networks.DeepSet(summary_dim=16,
                                    mlp_widths_invariant_outer=(64,4))
    inf_net_ds_il = bf.networks.CouplingFlow(transform="spline")

    workflow_ds_il = bf.BasicWorkflow(
        simulator=sim,
        adapter=adapter,
        summary_network=sum_net_ds_il,
        inference_conditions=inf_net_ds_il
    ) 

    sum_net_ds_mha = DeepSetMHA(summary_dim=16, mlp_widths_equivariant=(64,64))
    inf_net_ds_mha = bf.networks.CouplingFlow(transform="spline")

    workflow_ds_mha = bf.BasicWorkflow(
        simulator=sim,
        adapter=adapter,
        summary_network=sum_net_ds_mha,
        inference_conditions=inf_net_ds_mha
    )

    #-- GCN --#
    sum_net_gcn_mp = GCNMeanPooling(summary_dim=16, gcn_units=(64,64))
    inf_net_gcn_mp = bf.networks.CouplingFlow(transform="spline")

    workflow_gcn_mp = bf.BasicWorkflow(
        simulator=sim,
        adapter=adapter,
        summary_network=sum_net_gcn_mp,
        inference_conditions=inf_net_gcn_mp
    )

    sum_net_gcn_il = GCNInvariantLayer(summary_dim=16, gcn_units=(64,64), mlp_widths_pooling=(64,4))
    inf_net_gcn_il = bf.networks.CouplingFlow(transform="spline")

    workflow_gcn_il = bf.BasicWorkflow(
        simulator=sim,
        adapter=adapter,
        summary_network=sum_net_gcn_il,
        inference_conditions=inf_net_gcn_il
    )

    sum_net_gcn_mha = GCNMHA(summary_dim=16, gcn_units=(64,64))
    inf_net_gcn_mha = bf.networks.CouplingFlow(transform="spline")

    workflow_gcn_mha = bf.BasicWorkflow(
        simulator=sim,
        adapter=adapter,
        summary_network=sum_net_gcn_mha,
        inference_conditions=inf_net_gcn_mha
    )

    #-- SetTransformer --#
    sum_net_st_mp = SetTransformerMeanPooling(summary_dim=16, embed_dims=(64,64))
    inf_net_st_mp = bf.networks.CouplingFlow(transform="spline")

    workflow_st_mp = bf.BasicWorkflow(
        simulator=sim,
        adapter=adapter,
        summary_network=sum_net_st_mp,
        inference_conditions=inf_net_st_mp
    )

    sum_net_st_il = SetTransformerInvariantLayer(summary_dim=16, embed_dims=(64,64), mlp_widths_pooling=(64,4))
    inf_net_st_il = bf.networks.CouplingFlow(transform="spline")

    workflow_st_il = bf.BasicWorkflow(
        simulator=sim,
        adapter=adapter,
        summary_network=sum_net_st_il,
        inference_conditions=inf_net_st_il
    )

    sum_net_st_mha = bf.networks.SetTransformer(summary_dim=16, embed_dims=(64,64))
    inf_net_st_mha = bf.networks.CouplingFlow(transform="spline")

    workflow_st_mha = bf.BasicWorkflow(
        simulator=sim,
        adapter=adapter,
        summary_network=sum_net_st_mha,
        inference_conditions=inf_net_st_mha
    )

    #-- GraphTransformer --#
    sum_net_gt_mp = GraphTransformerMeanPooling(summary_dim=16, embed_dims=(64,64), mlp_widths_pooling=(64,4))
    inf_net_gt_mp = bf.networks.CouplingFlow(transform="spline")

    workflow_gt_mp = bf.BasicWorkflow(
        simulator=sim,
        adapter=adapter,
        summary_network=sum_net_gt_mp,
        inference_conditions=inf_net_gt_mp
    )

    
    sum_net_gt_il = GraphTransformerInvariantLayer(summary_dim=16, embed_dims=(64,64))
    inf_net_gt_il = bf.networks.CouplingFlow(transform="spline")

    workflow_gt_il = bf.BasicWorkflow(
        simulator=sim,
        adapter=adapter,
        summary_network=sum_net_gt_il,
        inference_conditions=inf_net_gt_il
    )

    sum_net_gt_mha = GraphTransformer(summary_dim=16, embed_dims=(64,64))
    inf_net_gt_mha = bf.networks.CouplingFlow(transform="spline")

    workflow_gt_mha = bf.BasicWorkflow(
        simulator=sim,
        adapter=adapter,
        summary_network=sum_net_gt_mha,
        inference_conditions=inf_net_gt_mha
    )


    workflows = [workflow_gcn_mp, workflow_gcn_il, workflow_gcn_mha, workflow_ds_mp, workflow_ds_il, workflow_ds_mha, 
             workflow_st_mp, workflow_st_il, workflow_st_mha, workflow_gt_mp, workflow_gt_il, workflow_gt_mha]

    wf_names = ["GCN-MeanPooling", "GCN-InvariantLayer", "GCN-MHAttention", "DeepSet-MeanPooling", "DeepSet-InvariantLayer", "DeepSet-MHAttention", 
            "SetTrans-MeanPooling", "SetTrans-InvariantLayer", "SetTrans-MHAttention","GraphTrans-MeanPooling", "GraphTrans-InvariantLayer", "GraphTrans-MHAttention"]
    

    for i in range(12):
        wf = workflows[i]
        wf_name = wf_names[i]

        hist = wf.fit_online(epochs)

        # save approximator
        filepath = Path("checkpoints/case_study_toy") / f"toy_example_{wf_name}_{run}.keras"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        workflows[i].approximator.save(filepath=str(filepath)) 

        # save last loss
        vl = (hist.history.get("val_loss") 
                  if "val_loss" in hist.history 
                  else hist.history.get("loss", []))
        last_loss = float(vl[-1]) if len(vl) else np.nan
        append_val_row(wf_name, run, last_loss, path=VAL_CSV)

        # compute and save metrics
        meta = {"run": str(run), "workflow": wf_name}
        m = compute_metrics_for_workflow(wf.approximator, test_data)
        row = {**meta, **m}
        append_row_atomic(CSV_PATH, row, columns=COLUMNS)





