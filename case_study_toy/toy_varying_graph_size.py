################################################################################
##                                                                            ##
##                      Case Study Toy: Varying graph size                    ##
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



MAX_NODES = 50

def prior():
    pi_aa = np.random.uniform(low=0.1, high=0.9)
    pi_bb = np.random.uniform(low=0.1, high=0.9)
    pi_ab = np.random.uniform(low=0.1, high=0.9)
    gamma = np.random.uniform(low=0.1, high=0.9)

    n_nodes = np.random.randint(low=10, high=MAX_NODES)
    if n_nodes == 10:
        num_a = 5
    else:
        num_a = np.random.randint(low=5, high=n_nodes-5)
    return dict(pi_aa=pi_aa, pi_bb=pi_bb, pi_ab=pi_ab, gamma=gamma, n_nodes=n_nodes, num_a=num_a)


def simulator(pi_aa, pi_bb, pi_ab, gamma, num_a, n_nodes):

    adj = np.zeros((MAX_NODES, MAX_NODES), dtype=np.int8)
    common = np.zeros((MAX_NODES, MAX_NODES), dtype=np.int8)
    x = np.zeros((MAX_NODES, 1), dtype=np.int8)
    x[:num_a, ] = 1  
    x[num_a:n_nodes] = -1 

    # base probabilities
    for i in range(n_nodes - 1):
        for j in range(i + 1, n_nodes):
            if x[i, 0] == x[j, 0]:
                p_edge = pi_aa if x[i, 0] == 1 else pi_bb
            else:
                p_edge = pi_ab

            adj[i, j] = np.random.binomial(1, p_edge)
            adj[j, i] = adj[i, j]

    # count common neighbors
    for i in range(n_nodes - 1):
        for j in range(i + 1, n_nodes):
            common[i, j] = np.sum(adj[i, :] * adj[j, :])
    
    # triadic closure
    for i in range(n_nodes - 1):
        for j in range(i + 1, n_nodes):
            if (adj[i, j] == 0) and (common[i, j] > 0):
                adj[i, j] = np.random.binomial(1, gamma)
                adj[j, i] = adj[i, j]

    # padding
    if(n_nodes < MAX_NODES):
        for i in range(MAX_NODES):
            for j in range(n_nodes, MAX_NODES):
                adj[i,j] = -1
                adj[j,i] = -1

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
      .concatenate(["n_nodes"], into="inference_conditions")   
)

sum_net_st_mha = bf.networks.SetTransformer(summary_dim=16, embed_dims=(64,64))
inf_net_st_mha = bf.networks.CouplingFlow(transform="spline")

workflow_st_mha = bf.BasicWorkflow(
    simulator=sim,
    adapter=adapter,
    summary_network=sum_net_st_mha,
    inference_conditions=inf_net_st_mha
)

hist = workflow_st_mha.fit_online(epochs = 250)


test_data = sim.sample(200)

workflow_st_mha.plot_default_diagnostics(test_data)

## -- Saving
filepath = Path("checkpoints/case_study_toy") / f"toy_example_SetTransformer_varying_nodes.keras"
filepath.parent.mkdir(parents=True, exist_ok=True)
workflow_st_mha.approximator.save(filepath=str(filepath)) 

## -- Load the trained approximator
filepath = Path("checkpoints/case_study_toy") / f"toy_example_SetTransformer_varying_nodes.keras"
approx = keras.saving.load_model(filepath)


test_data = sim.sample(200)
post_draws = approx.sample(conditions=test_data, num_samples =500)

bf.diagnostics.recovery(targets=test_data, estimates=post_draws)
bf.diagnostics.calibration_ecdf(targets=test_data, estimates=post_draws, difference=True)

## diagnostics for different sizes of graphs
def make_fixed_N_recovery_plot(N=10, n_samples = 200):
    def prior_N():
        pi_aa = np.random.uniform(low=0.1, high=0.9)
        pi_bb = np.random.uniform(low=0.1, high=0.9)
        pi_ab = np.random.uniform(low=0.1, high=0.9)
        gamma = np.random.uniform(low=0.1, high=0.9)

        n_nodes = N
        if n_nodes == 10:
            num_a = 5
        else:
            num_a = np.random.randint(low=5, high=n_nodes-5)
        return dict(pi_aa=pi_aa, pi_bb=pi_bb, pi_ab=pi_ab, gamma=gamma, n_nodes=n_nodes, num_a=num_a)

    sim_N = bf.make_simulator([prior_N, simulator])
    test_data_N = sim_N.sample(n_samples)
    post_draws_N = approx.sample(conditions=test_data_N, num_samples=500)

    bf.diagnostics.recovery(targets=test_data_N, estimates=post_draws_N, variable_keys=["pi_ab", "gamma"],
    variable_names=[r"$\pi_{AB}$", r"$\lambda$"], label_fontsize=20, title_fontsize=30, tick_fontsize=16, metric_fontsize=20)
    fig = plt.gcf()
    fig.suptitle(f"N ={N}", fontsize=30)


make_fixed_N_recovery_plot(N=45, n_samples=500)





def prior_15():
    pi_aa = np.random.uniform(low=0.1, high=0.9)
    pi_bb = np.random.uniform(low=0.1, high=0.9)
    pi_ab = np.random.uniform(low=0.1, high=0.9)
    gamma = np.random.uniform(low=0.1, high=0.9)

    n_nodes = 15
    if n_nodes == 10:
        num_a = 5
    else:
        num_a = np.random.randint(low=5, high=n_nodes-5)
    return dict(pi_aa=pi_aa, pi_bb=pi_bb, pi_ab=pi_ab, gamma=gamma, n_nodes=n_nodes, num_a=num_a)



sim_15 = bf.make_simulator([prior_15, simulator])
test_data_15 = sim_15.sample(200)
post_draws_15 = approx.sample(conditions=test_data_15, num_samples=500)

bf.diagnostics.recovery(targets=test_data_15, estimates=post_draws_15, variable_keys=["pi_ab", "gamma"],
variable_names=[r"$\pi_{AB}$", r"$\lambda$"], label_fontsize=20, title_fontsize=30, tick_fontsize=16, metric_fontsize=20)
fig = plt.gcf()
fig.suptitle("N = 15", fontsize=30)
plt.savefig("plots/toy_example_varyN_15_recovery.pdf")


def prior_45():
    pi_aa = np.random.uniform(low=0.1, high=0.9)
    pi_bb = np.random.uniform(low=0.1, high=0.9)
    pi_ab = np.random.uniform(low=0.1, high=0.9)
    gamma = np.random.uniform(low=0.1, high=0.9)

    n_nodes = 45
    if n_nodes == 10:
        num_a = 5
    else:
        num_a = np.random.randint(low=5, high=n_nodes-5)
    return dict(pi_aa=pi_aa, pi_bb=pi_bb, pi_ab=pi_ab, gamma=gamma, n_nodes=n_nodes, num_a=num_a)

sim_45 = bf.make_simulator([prior_45, simulator])
test_data_45 = sim_45.sample(200)
post_draws_45 = approx.sample(conditions=test_data_45, num_samples=500)

bf.diagnostics.recovery(targets=test_data_45, estimates=post_draws_45, variable_keys=["pi_ab", "gamma"],
variable_names=[r"$\pi_{AB}$", r"$\lambda$"], label_fontsize=20, title_fontsize=30, tick_fontsize=16, metric_fontsize=20)
fig = plt.gcf()
fig.suptitle("N = 45", fontsize=30)
plt.savefig("plots/toy_example_varyN_45_recovery.pdf")

