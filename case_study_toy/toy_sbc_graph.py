################################################################################
##                                                                            ##
##                 Case Study Toy: Calibration of graph metrics               ##
##                                                                            ##
################################################################################


from pathlib import Path
import numpy as np
import pandas as pd
import bayesflow as bf
import keras
import matplotlib.pyplot as plt
import csv
import random

#-------------------------- Prior and Simulator --------------------------#
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

#------------ Sample test data and corresponding posterior draws --------------#
random.seed(31415)
test_data = sim.sample(250)

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
            draw = {k: float(post_draws[k][b, s, 0]) for k in ["pi_aa", "pi_bb", "pi_ab", "gamma"]}
            
            # num_a must be int; if it's in post_draws use it, otherwise infer from conditions
            num_a = int(round(float(post_draws["num_a"][b, s, 0])))

            out = simulator(
                pi_aa  = draw["pi_aa"],
                pi_bb  = draw["pi_bb"],
                pi_ab  = draw["pi_ab"],
                gamma  = draw["gamma"],
                num_a  = num_a,
                N_NODES= N_NODES,
            )
            graphs[b, s] = out["graph"]   # (N, N+1)

        if (b + 1) % 100 == 0:
            print(f"  {b + 1}/{B} done")

    result = dict(post_draws)
    result["graph"] = graphs   # (B, S, N, N+1)
    return result

#------------------------------ Graph metrics ---------------------------------#


def spectral_gap(data):
    """
    Vectorized spectral gap over a batch B.

    Spectral gap = lambda_2 - lambda_1 of the normalized Laplacian,
    where lambda_1 = 0 (for connected graphs) and lambda_2 is the
    Fiedler value — larger means better connected / more separated communities.

    Expects:
      data["graph"]  shape (B, N, 1+N) with x in [:,:,0] and adj in [:,:,-N:]

    Returns:
      sg: shape (B,) float spectral gaps, one per graph
    """
    graph = data["graph"]
    adj   = graph[:, :, -N_NODES:].astype(np.float32)   # (B, N, N)

    B = adj.shape[0]
    sg = np.zeros(B, dtype=np.float32)

    for b in range(B):
        A   = adj[b]                                     # (N, N)
        deg = A.sum(axis=1)                              # (N,)

        # normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg + 1e-12), 0.0)
        S = np.diag(inv_sqrt)
        L = np.eye(N_NODES) - S @ A @ S

        evals = np.linalg.eigvalsh(L)                   # sorted ascending
        evals = np.sort(evals)

        # lambda_1 ~ 0, spectral gap = lambda_2 - lambda_1
        sg[b] = evals[1] - evals[0]

    return sg

def edge_density(data):
    """
    Vectorized edge density over a batch B.

    Edge density = actual edges / possible edges
                 = sum(A) / (N * (N-1))
    (upper triangle only, undirected graph)

    Expects:
      data["graph"]  shape (B, N, 1+N) with x in [:,:,0] and adj in [:,:,-N:]

    Returns:
      ed: shape (B,) float edge densities, one per graph
    """
    graph = data["graph"]
    adj   = graph[:, :, -N_NODES:].astype(np.float32)   # (B, N, N)

    # sum upper triangle only (undirected, no self-loops)
    idx           = np.triu_indices(N_NODES, k=1)
    upper_sum     = adj[:, idx[0], idx[1]].sum(axis=1)  # (B,)
    possible_edges = N_NODES * (N_NODES - 1) / 2

    return (upper_sum / possible_edges).astype(np.float32)

def degree_assortativity(data):
    """
    Vectorized degree assortativity over a batch B.

    Pearson correlation between degrees of connected node pairs (i, j):
      r = (M^{-1} sum_{ij} j_i j_j - [M^{-1} sum_{ij} 0.5(j_i + j_j)]^2) /
          (M^{-1} sum_{ij} 0.5(j_i^2 + j_j^2) - [M^{-1} sum_{ij} 0.5(j_i + j_j)]^2)
    where the sums run over edges and M is the number of edges.

    Returns 0 for graphs with no edges or zero variance in degree.

    Expects:
      data["graph"]  shape (B, N, 1+N) with x in [:,:,0] and adj in [:,:,-N:]

    Returns:
      da: shape (B,) float assortativity coefficients in [-1, 1], one per graph
    """
    graph = data["graph"]
    adj   = graph[:, :, -N_NODES:].astype(np.float32)   # (B, N, N)

    B  = adj.shape[0]
    da = np.zeros(B, dtype=np.float32)

    idx = np.triu_indices(N_NODES, k=1)                  # upper triangle indices

    for b in range(B):
        A    = adj[b]
        deg  = A.sum(axis=1)                             # (N,)
        edge_mask = A[idx[0], idx[1]] > 0               # which pairs are connected

        if edge_mask.sum() == 0:
            continue

        di = deg[idx[0]][edge_mask]                      # degree of node i per edge
        dj = deg[idx[1]][edge_mask]                      # degree of node j per edge

        M    = len(di)
        mean = (di + dj).sum() / (2 * M)
        num  = (di * dj).sum() / M - mean ** 2
        den  = (0.5 * (di**2 + dj**2)).sum() / M - mean ** 2

        da[b] = num / den if den > 1e-12 else 0.0

    return da

def global_clustering_coefficient(data):
    """
    Vectorized global clustering coefficient (transitivity) over a batch B.

    Transitivity = 3 * number of triangles / number of connected triplets
                 = trace(A^3) / (sum(A^2) - trace(A^2))

    A value of 1 means every connected triplet closes into a triangle,
    0 means no triangles exist.

    Expects:
      data["graph"]  shape (B, N, 1+N) with x in [:,:,0] and adj in [:,:,-N:]

    Returns:
      gcc: shape (B,) float clustering coefficients in [0, 1], one per graph
    """
    graph = data["graph"]
    adj   = graph[:, :, -N_NODES:].astype(np.float32)   # (B, N, N)

    A2 = np.einsum("bij,bjk->bik", adj, adj)             # (B, N, N)
    A3 = np.einsum("bij,bjk->bik", A2,  adj)             # (B, N, N)

    triangles = np.einsum("bii->b", A3)                  # trace(A^3) = 6 * n_triangles
    triplets  = A2.sum(axis=(1, 2)) - np.einsum("bii->b", A2)  # sum(A^2) - trace(A^2)

    gcc = np.where(triplets > 0, triangles / triplets, 0.0)

    return gcc.astype(np.float32)


test_quantities={r"Spectral Gap": spectral_gap,
                    r"Edge Density": edge_density, 
                    r"Degree Assortativity" : degree_assortativity,
                    r"Global Clustering Coefficient": global_clustering_coefficient}


#----------------------- Run for all approximators ----------------------------#

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

num_samples = 250
num_a = np.repeat(test_data["num_a"][:, :, np.newaxis], num_samples, axis=1)   


wf_names = ["GCN-MeanPooling", "GCN-InvariantLayer", "GCN-MHAttention", "DeepSet-MeanPooling", "DeepSet-InvariantLayer", "DeepSet-MHAttention", 
            "SetTrans-MeanPooling", "SetTrans-InvariantLayer", "SetTrans-MHAttention","GraphTrans-MeanPooling", "GraphTrans-InvariantLayer", "GraphTrans-MHAttention"]

runs = [1,2,3,4,5]

metric_names = ["spectral_gap", "edge_density", "degree_assortativity", "global_clustering"]

rows = []

output_path = Path("case_study_toy/results") / "SBC_data.csv"

with open(output_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["wf_name", "run", "metric", "lg_val", "r_val"])
    writer.writeheader()

    for i in range(12):
        wf_name = wf_names[i]
        for run in runs:
            filepath = Path("checkpoints/case_study_toy") / f"toy_example_{wf_name}_{run}.keras"
            app = keras.saving.load_model(filepath=filepath)

            post_draws = app.sample(conditions=test_data, num_samples=num_samples)
            post_draws["num_a"] = num_a
            post_draws_with_graphs = simulate_posterior_predictive_graphs(post_draws, simulator)

            lg = bf.diagnostics.metrics.calibration_log_gamma(estimates=post_draws_with_graphs, targets=test_data,
                                  variable_keys=[], test_quantities=test_quantities)
            lg_vals = np.asarray(lg.get("values", lg)).ravel()

            updated_data = bf.utils.dict_utils.compute_test_quantities(
                targets=test_data,
                estimates=post_draws_with_graphs,
                variable_keys=[],
                variable_names=[],
                test_quantities=test_quantities,
            )
            plot_data = bf.utils.plot_utils.prepare_plot_data(
                estimates=updated_data["estimates"],
                targets=updated_data["targets"],
                variable_keys=updated_data["variable_keys"],
                variable_names=updated_data["variable_names"],
                num_col=None,
                num_row=None,
                figsize=None,
            )

            estimates = plot_data.pop("estimates")
            targets   = plot_data.pop("targets")
            point_est = np.median(estimates, axis=1)

            if point_est.shape[0] != targets.shape[0] and point_est.shape[1] == targets.shape[0]:
                point_est = point_est.T

            n_params = min(targets.shape[1], point_est.shape[1], 4)
            r_vals = []
            for j in range(n_params):
                t = targets[:, j]
                p = point_est[:, j]
                if np.std(t) == 0 or np.std(p) == 0:
                    r_vals.append(np.nan)
                else:
                    r_vals.append(float(np.corrcoef(t, p)[0, 1]))
            while len(r_vals) < 4:
                r_vals.append(np.nan)

            # Write rows immediately after each model is done
            for k, metric in enumerate(metric_names):
                writer.writerow({
                    "wf_name": wf_name,
                    "run":     run,
                    "metric":  metric,
                    "lg_val":  float(lg_vals[k]) if k < len(lg_vals) else np.nan,
                    "r_val":   float(r_vals[k]),
                })
            f.flush()  # force write to disk immediately
            print(f"Saved {wf_name} run {run}")








