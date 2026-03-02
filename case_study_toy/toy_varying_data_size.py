################################################################################
##                                                                            ##
##                  Case Study Toy: Different sizes of datasets               ##
##                                                                            ##
################################################################################

import numpy as np
import pandas as pd
import keras
import bayesflow as bf
from pathlib import Path
import random
import time
from scipy import stats
import json

from summary_networks.GCNInvariantLayer import GCNInvariantLayer
from summary_networks.GraphTransformerAttentionPooling import GraphTransformer


### --- Define Simulator --- ###
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


epochs = 100

### --- Sample the training data for different data sizes --- ###

random.seed(31415)
dataset_3200 = sim.sample(3200)
dataset_6400 = sim.sample(6400)
dataset_12800 = sim.sample(12800)    
dataset_32000 = sim.sample(32000)    
val_data = sim.sample(3200)

data = [dataset_3200, dataset_6400, dataset_12800, dataset_32000]
datasizes = [3200, 6400, 12800, 32000]



runtimes = {}
losses = {}

### --- Running the simulation --- ###

for i in range(4):

    #-- networks --#
    sum_net_gt_mha = GraphTransformer(summary_dim=16, embed_dims=(64,64))
    inf_net_gt_mha = bf.networks.CouplingFlow(transform="spline")

    workflow_gt_mha = bf.BasicWorkflow(
        simulator=sim,
            adapter=adapter,
            summary_network=sum_net_gt_mha,
            inference_conditions=inf_net_gt_mha
    )

    sum_net_st_mha = bf.networks.SetTransformer(summary_dim=16, embed_dims=(64,64))
    inf_net_st_mha = bf.networks.CouplingFlow(transform="spline")

    workflow_st_mha = bf.BasicWorkflow(
        simulator=sim,
        adapter=adapter,
        summary_network=sum_net_st_mha,
        inference_conditions=inf_net_st_mha
    )

    sum_net_gcn_il = GCNInvariantLayer(summary_dim=16, gcn_units=(64,64), mlp_widths_pooling=(64,4))
    inf_net_gcn_il = bf.networks.CouplingFlow(transform="spline")

    workflow_gcn_il = bf.BasicWorkflow(
        simulator=sim,
        adapter=adapter,
        summary_network=sum_net_gcn_il,
        inference_conditions=inf_net_gcn_il
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

    workflows = [workflow_gcn_il, workflow_ds_il, workflow_st_mha, workflow_gt_mha]

    wf_names = ["GCN-InvariantLayer", "DeepSet-InvariantLayer", "SetTrans-MHAttention", "GraphTrans-MHAttention"]

    for j in range(1):
        
        workflow = workflows[j]

        start_time = time.time()
        hist = workflow.fit_offline(data = data[i], batch_size=32, epochs=epochs, validation_data=val_data)
        end_time = time.time()

        key = f"{wf_names[j]}_{datasizes[i]}"
        runtimes[key] = end_time - start_time
        losses[key]   = {
            "train_loss": float(hist.history["loss"][-1]),
            "val_loss":   float(hist.history["val_loss"][-1]),
        }
        
        filepath = Path("checkpoints/case_study_toy") / f"toy_example_{wf_names[j]}_datasize_{datasizes[i]}.keras"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        workflow.approximator.save(filepath=str(filepath)) 

### --- Save the runtime and losses --- ###
with open("case_study_toy/results/runtimes.json", "w") as f:
    json.dump(runtimes, f, indent=2)


with open("case_study_toy/results/losses.json", "w") as f:
    json.dump(losses, f, indent=2)


### --- Evaluate runtime --- ###
# convert runtimes dict to dataframe
rows_rt = []
for key, runtime in runtimes.items():
    network, datasize = key.rsplit("_", 1)
    rows_rt.append({
        "Network":      network,
        "Dataset Size": int(datasize),
        "Runtime (min)": round(runtime / 60, 2)
    })

df_runtimes = pd.DataFrame(rows_rt)

# pivot into 4x4 matrix
network_order = ["DeepSet-InvariantLayer", "GCN-InvariantLayer", 
                 "GraphTrans-MHAttention", "SetTrans-MHAttention"]

df_rt_pivot = (df_runtimes
               .pivot(index="Dataset Size", columns="Network", values="Runtime (min)")
               .round(2)[network_order]
               .sort_index())




### --- Calculate the recovery --- ###
random.seed(31415)
test_data = sim.sample(500)  

wf_names = ["GCN-InvariantLayer", "DeepSet-InvariantLayer", "SetTrans-MHAttention", "GraphTrans-MHAttention"]
rows = []
datasizes = [3200, 6400, 12800, 32000]
for i, datasize in enumerate(datasizes):
    for wf_name in wf_names:

        key      = f"{wf_name}_{datasize}"
        filepath = Path("checkpoints/case_study_toy") / f"toy_example_{wf_name}_datasize_{datasize}.keras"

        # load approximator and sample
        approximator = keras.saving.load_model(str(filepath))
        samples      = approximator.sample(conditions=test_data, num_samples=500) 

        row = {"Network": wf_name, "Dataset Size": datasize}

        for param in ["pi_aa", "pi_bb", "pi_ab", "gamma"]:
            draws     = samples[param]
            true_vals = test_data[param].squeeze()

            # recovery: correlation of posterior median to true value
            post_median = np.median(draws, axis=1).squeeze()
            r, _        = stats.pearsonr(post_median, true_vals)



            row[f"{param}_correlation"] = round(float(r),  4)

        rows.append(row)

df = pd.DataFrame(rows).sort_values(["Dataset Size", "Network"]).reset_index(drop=True)
df.to_csv("case_study_toy/results/recovery_varying_datasizes.csv", index=False)


