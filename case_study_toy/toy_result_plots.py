################################################################################
##                                                                            ##
##                           Case Study Toy: Results                          ##
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

### --- Simulator --- ###
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


### --- Plotting for Paper --- ###
random.seed(31415)
test_data = sim.sample(500)
test_data_small = sim.sample(100)


## -- GCN
from summary_networks.GCNMeanPooling import GCNMeanPooling
wf_name = "GCN-MeanPooling"
run = 1

file = Path("checkpoints/case_study_toy") / f"toy_example_{wf_name}_{run}.keras"
app_gcn = keras.saving.load_model(file)

post_draws_gcn = app_gcn.sample(num_samples=500, conditions=test_data)
post_draws_gcn_small = app_gcn.sample(num_samples=500, conditions=test_data_small)

f = bf.diagnostics.calibration_ecdf(targets=test_data, estimates=post_draws_gcn, difference=True, stacked=True,
        variable_keys=["pi_aa","pi_ab","pi_bb"], variable_names=[r"$\pi_{AB}$",r"$\pi_{AB}$",r"$\pi_{AB}$"], 
        legend_location=None, figsize=[4.8,4.8],
        label_fontsize=22, title_fontsize=30, tick_fontsize=16, metric_fontsize=22)
plt.title(r"$\pi_{AA},\pi_{BB},\pi_{AB}$", fontsize = 30)
plt.legend().remove()
#f.savefig("plots\\toy_example_calibration_pi_gcn.pdf")

f = bf.diagnostics.calibration_ecdf(targets=test_data, estimates=post_draws_gcn, difference=True, 
        variable_keys=["gamma"], variable_names=[r"$\lambda$"], figsize=[4.8,4.8], legend_location=None,
        label_fontsize=22, title_fontsize=30, tick_fontsize=16, metric_fontsize=22)
plt.ylabel("")
plt.legend().remove()
#f.savefig("plots\\toy_example_calibration_lambda_gcn.pdf")

f = bf.diagnostics.recovery(targets=test_data_small, estimates=post_draws_gcn_small, variable_keys=["pi_ab", "gamma"],
    variable_names=["$\pi_{AB}$", "$\lambda$"], figsize=[9.6,4.8],
    label_fontsize=22, title_fontsize=30, tick_fontsize=16, metric_fontsize=22)
#f.savefig("plots\\toy_example_recovery_gcn.pdf")


## -- SetTransformer
wf_name = "SetTrans-MHAttention"
run = 1

file = Path("checkpoints/case_study_toy") / f"toy_example_{wf_name}_{run}.keras"
app_st = keras.saving.load_model(file)

post_draws_st = app_st.sample(num_samples=500, conditions=test_data)
post_draws_st_small = app_st.sample(num_samples=500, conditions=test_data_small)

f = bf.diagnostics.calibration_ecdf(targets=test_data, estimates=post_draws_st, difference=True, stacked=True,
        variable_keys=["pi_aa","pi_ab","pi_bb"], variable_names=[r"$\pi_{AB}$",r"$\pi_{AB}$",r"$\pi_{AB}$"], 
        legend_location=None, figsize=[4.8,4.8],
        label_fontsize=22, title_fontsize=30, tick_fontsize=16, metric_fontsize=22)
plt.title(r"$\pi_{AA},\pi_{BB},\pi_{AB}$", fontsize = 30)
plt.legend().remove()
#f.savefig("plots\\toy_example_calibration_pi_st.pdf")

f = bf.diagnostics.calibration_ecdf(targets=test_data, estimates=post_draws_st, difference=True, 
        variable_keys=["gamma"], variable_names=[r"$\lambda$"], figsize=[4.8,4.8], legend_location="lower right",
        label_fontsize=22, title_fontsize=30, tick_fontsize=16, metric_fontsize=22)
plt.ylabel("")
#f.savefig("plots\\toy_example_calibration_lambda_st.pdf")

f = bf.diagnostics.recovery(targets=test_data_small, estimates=post_draws_st_small, variable_keys=["pi_ab", "gamma"],
    variable_names=["$\pi_{AB}$", "$\lambda$"], figsize=[9.6,4.8],
    label_fontsize=22, title_fontsize=30, tick_fontsize=16, metric_fontsize=22)
#f.savefig("plots\\toy_example_recovery_st.pdf")
