################################################################################
##                                                                            ##
##                   Case Study Toy: Analytical likelihood case               ##
##                                                                            ##
################################################################################

import numpy as np
import pandas as pd
import keras
import bayesflow as bf
from pathlib import Path
import matplotlib.pyplot as plt
import random
import json
import time

### ---- Simulator and Priors --- ###

N_NODES = 30

def prior():
    pi_aa = np.random.uniform(low=0.1, high=0.9)
    pi_bb = np.random.uniform(low=0.1, high=0.9)
    pi_ab = np.random.uniform(low=0.1, high=0.9)

    num_a = np.random.randint(low=5, high=25)
    return dict(pi_aa=pi_aa, pi_bb=pi_bb, pi_ab=pi_ab, num_a=num_a)



def simulator(pi_aa, pi_bb, pi_ab, num_a, N_NODES = N_NODES):

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

    graph_tensor = np.concatenate([x, adj.astype(np.float32)], axis=-1)
    return {"graph": graph_tensor}



sim = bf.make_simulator([prior, simulator])

adapter = (
    bf.Adapter()
      .constrain("pi_aa", lower=0.1, upper=0.9)
      .constrain("pi_bb", lower=0.1, upper=0.9)
      .constrain("pi_ab", lower=0.1, upper=0.9)

      .concatenate(["pi_aa", "pi_bb", "pi_ab"], into="inference_variables")  
      .rename("graph", "summary_variables")  
)

### ---- Fit Set Transformer --- ###
sum_net_st_mha = bf.networks.SetTransformer(summary_dim=16, embed_dims=(64,64))
inf_net_st_mha = bf.networks.CouplingFlow(transform="spline")

workflow_st_mha = bf.BasicWorkflow(
    simulator=sim,
    adapter=adapter,
    summary_network=sum_net_st_mha,
    inference_conditions=inf_net_st_mha
)

t_start = time.time()
hist = workflow_st_mha.fit_online(epochs=150)
t_end = time.time()
training_time = t_end - t_start

test_data = sim.sample(100)
workflow_st_mha.plot_default_diagnostics(test_data)

filepath = Path("checkpoints/case_study_toy") / f"analyitcal_likelihood_st.keras"
filepath.parent.mkdir(parents=True, exist_ok=True)
workflow_st_mha.approximator.save(filepath=str(filepath))

### ---- SBC  --- ###

filepath = Path("checkpoints/case_study_toy") / f"analyitcal_likelihood_st.keras"
app = keras.saving.load_model(filepath)

## Load the datasets
with open("case_study_toy/analytical_likelihood_case/results/sbc_datasets.json", "r") as f:
    raw = json.load(f)

n_sims  = len(raw)
n_nodes = len(raw[0]["x"])


graphs = np.zeros((n_sims, n_nodes, n_nodes + 1), dtype=np.float32)
for i, d in enumerate(raw):
    x   = np.array(d["x"], dtype=np.float32).reshape(-1, 1)   # (N, 1)
    adj = np.array(d["adj"], dtype=np.float32)                 # (N, N)
    graphs[i] = np.concatenate([x, adj], axis=-1)

start_time = time.time()
posterior_draws = app.sample(conditions={"graph":graphs}, num_samples=500)
end_time = time.time()
end_time - start_time

theta_true = {
    "pi_aa": np.array([d["pi_aa"] for d in raw], dtype=np.float32).reshape(-1, 1),
    "pi_bb": np.array([d["pi_bb"] for d in raw], dtype=np.float32).reshape(-1, 1),
    "pi_ab": np.array([d["pi_ab"] for d in raw], dtype=np.float32).reshape(-1, 1),
}

true_draws = theta_true
true_draws["graph"] = graphs


### ---- SBC of likelihood  --- ###

def simulate_posterior_predictive_graphs(post_draws, simulator, N_NODES=N_NODES):
    """
    For each (b, s) in (B=100, S=500), simulate a graph from the posterior draw.
    
    Expects post_draws dict with keys pi_aa, pi_bb, pi_ab, gamma, each (B, S, 1).
    Returns the same dict with an added "graph" entry of shape (B, S, N, N+1).
    """
    B, S, _ = post_draws["pi_aa"].shape
    
    # N+1 because simulator returns graph with x concatenated (shape N x N+1)
    graphs = np.empty((B, S, N_NODES, N_NODES + 1), dtype=np.float32)

    for b in range(B):
        for s in range(S):
            draw = {k: float(post_draws[k][b, s, 0]) for k in ["pi_aa", "pi_bb", "pi_ab"]}
            
            # num_a must be int; if it's in post_draws use it, otherwise infer from conditions
            num_a = int(round(float(post_draws["num_a"][b, s, 0])))

            out = simulator(
                pi_aa  = draw["pi_aa"],
                pi_bb  = draw["pi_bb"],
                pi_ab  = draw["pi_ab"],
                num_a  = num_a,
                N_NODES= N_NODES,
            )
            graphs[b, s] = out["graph"]   # (N, N+1)

        if (b + 1) % 100 == 0:
            print(f"  {b + 1}/{B} done")

    result = dict(post_draws)
    result["graph"] = graphs   # (B, S, N, N+1)
    return result


def log_likelihood(data):
    """
    Vectorized log-likelihood of the SBM over a batch B.

    For each edge (i,j) with i < j, the probability is:
      pi_aa  if x[i] == x[j] == 1
      pi_bb  if x[i] == x[j] == 0
      pi_ab  if x[i] != x[j]

    log p(adj | x, pi_aa, pi_bb, pi_ab)
      = sum_{i<j} adj[i,j] * log(p[i,j]) + (1 - adj[i,j]) * log(1 - p[i,j])

    Expects:
      data["graph"]    shape (B, N, 1+N) with x in [:,:,0] and adj in [:,:,1:]
      data["pi_aa"]  shape (B, 1) or (B,)
      data["pi_bb"]  shape (B, 1) or (B,)
      data["pi_ab"]  shape (B, 1) or (B,)

    Returns:
      ll: shape (B,) float log-likelihoods, one per graph
    """
    graph = data["graph"]
    x     = graph[:, :, 0]                              # (B, N)
    adj   = graph[:, :, 1:].astype(np.float32)          # (B, N, N)

    pi_aa = np.array(data["pi_aa"]).reshape(-1, 1, 1) # (B, 1, 1)
    pi_bb = np.array(data["pi_bb"]).reshape(-1, 1, 1)
    pi_ab = np.array(data["pi_ab"]).reshape(-1, 1, 1)

    # outer products to get pairwise group indicators  (B, N, N)
    x_i   = x[:, :, np.newaxis]                         # (B, N, 1)
    x_j   = x[:, np.newaxis, :]                         # (B, 1, N)

    same  = (x_i == x_j).astype(np.float32)             # (B, N, N)
    both1 = (x_i * x_j).astype(np.float32)              # (B, N, N) 1 iff both in group A

    # probability matrix  (B, N, N)
    p = both1 * pi_aa + same * (1 - both1) * pi_bb + (1 - same) * pi_ab

    # upper triangle mask — only i < j
    mask = np.triu(np.ones((N_NODES, N_NODES), dtype=bool), k=1)  # (N, N)

    p_upper   = p[:, mask]                               # (B, N*(N-1)/2)
    adj_upper = adj[:, mask]                             # (B, N*(N-1)/2)

    ll = (adj_upper * np.log(p_upper) +
          (1 - adj_upper) * np.log(1 - p_upper)).sum(axis=1)  # (B,)

    return ll.astype(np.float32)


num_a = num_a = graphs[:, :, 0].sum(axis=1, keepdims=True).astype(np.int32)  # (B,)
num_a = np.repeat(num_a, 500, axis=1)  
posterior_draws["num_a"] = np.expand_dims(num_a, axis=-1)

posterior_draws_with_graph = simulate_posterior_predictive_graphs(post_draws=posterior_draws, 
                        simulator=simulator, N_NODES=N_NODES)

bf.diagnostics.calibration_ecdf(estimates=posterior_draws_with_graph, targets=true_draws,
        variable_keys=["pi_aa", "pi_bb", "pi_ab"], variable_names=[r"$\pi_{AA}$",r"$\pi_{BB}$",r"$\pi_{AB}$"],
        test_quantities={r"Log-Likelihood": log_likelihood})
plt.suptitle(r"BayesFlow: Set Transformer", fontsize = 30)
#plt.savefig("plots\\toy_analytical_likelihood_ecdf_bf.pdf")


### ---- MCMC posterior vs bf samples  --- ###

with open("case_study_toy/analytical_likelihood_case/results/mcmc_posterior_stats.json", "r") as f:
    raw = json.load(f)

param_names  = ["pi_aa", "pi_bb", "pi_ab"]
param_labels = [r"$\pi_{aa}$", r"$\pi_{bb}$", r"$\pi_{ab}$"]

mcmc_median = {k: np.array(raw["median"][k], dtype=np.float32).reshape(-1) for k in param_names}
mcmc_q5   = {k: np.array(raw["q5"][k],   dtype=np.float32).reshape(-1) for k in param_names}
mcmc_q95  = {k: np.array(raw["q95"][k],  dtype=np.float32).reshape(-1) for k in param_names}

# BayesFlow metrics
bf_median = {k: np.median(posterior_draws[k],  axis=1).squeeze() for k in param_names}
bf_q5   = {k: np.percentile(posterior_draws[k], 5,  axis=1).squeeze() for k in param_names}
bf_q95  = {k: np.percentile(posterior_draws[k], 95, axis=1).squeeze() for k in param_names}

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

random.seed(31415)
for ax, name, label in zip(axes, param_names, param_labels):
    x = mcmc_median[name]
    y = bf_median[name]

    # randomly select half the points
    idx = np.random.choice(len(x), size=len(x) // 2, replace=False)
    x, y = x[idx], y[idx]

    # points and errorbar
    xerr = np.array([x - mcmc_q5[name][idx], mcmc_q95[name][idx] - x])
    yerr = np.array([y - bf_q5[name][idx],   bf_q95[name][idx]   - y])

    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", alpha=0.5,
                color="#132a70", elinewidth=0.5, markersize=2)

    # identity line
    all_vals = np.concatenate([x, y])
    lims = [all_vals.min() - 0.05, all_vals.max() + 0.05]
    ax.plot(lims, lims, color="black", alpha=0.9, linestyle="dashed")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel(f"MCMC ({label})")
    ax.set_ylabel(f"BayesFlow ({label})")
    ax.set_title(label)

    corr = np.corrcoef(mcmc_median[name], bf_median[name])[0, 1]
    ax.text(
        0.1, 0.9,
        f"$r$ = {corr:.3f}",
        transform=ax.transAxes,
        fontsize=12,
        ha="left"
    )

plt.tight_layout()
#plt.savefig("plots/toy_analytical_likelihood_recovery_bf_vs_mcmc.pdf")






