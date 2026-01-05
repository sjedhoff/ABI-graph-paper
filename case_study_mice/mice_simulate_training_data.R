################################################################################
##                                                                            ##
##          Case Study Mice: Generate Training Data for Bayesflow             ##
##                                                                            ##
################################################################################

# load the code from the simulator
source("case_study_mice/mice_simulation_helper_functions.R")


# fixed hyper-parameters
n_mice <- 30
total_taxa <- 20
active_taxa <- 5
days_to_simulate <- 40
extinction_threshold <- 0.01
presence_threshold <- 0.01
structure <- "community"

# set priors for parameters
prior <- function(n){
  network_density <- runif(n, min = 0.01, max = 0.5)
  exchange_factor <- runif(n, min = 0.05, max = 0.5)

  return(data.frame("exchange_factor" = exchange_factor,
                    "network_density" = network_density))
}


# shorter version of the simulator - no calculation of jaccard similarity
run_microbiome_simulation_short <- function(n_mice = 20, 
                                            total_taxa = 1000, 
                                            active_taxa = 300,
                                            days_to_simulate = 30,
                                            exchange_factor = 0.001,
                                            network_density = 0.2,
                                            extinction_threshold = 0.001,
                                            presence_threshold = 0.001,
                                            structure = "random") {
  
  

  
  # 1. generate network
  adj_matrix <- generate_weighted_adjacency_matrix(n_mice, structure, density = network_density)
  mouse_ids <- paste0("Mouse_", 1:n_mice)
  mice_pairs_df <- create_mice_pair_dataframe(adj_matrix, id_prefix = "Mouse")

  
  # 2. initialize microbiomes
  initial_microbiomes <- initialize_microbiomes(n_mice, total_taxa, active_taxa)

  
  # 3. run simulation with competition & threshold
  microbiome_history <- simulate_competitive_microbiome_exchange(
    adj_matrix,
    initial_microbiomes,
    days = days_to_simulate,
    exchange_factor = exchange_factor,
    extinction_threshold = extinction_threshold
  )

  
  # 4. return results
  return(list(
    parameters = list(
      n_mice = n_mice,
      total_taxa = total_taxa,
      active_taxa = active_taxa,
      days_simulated = days_to_simulate,
      exchange_factor = exchange_factor,
      network_density = network_density,
      extinction_threshold = extinction_threshold,
      presence_threshold = presence_threshold
    ),
    adjacency_matrix    = adj_matrix,
    microbiome_history  = microbiome_history
  ))
}




# helper function
simulator <- function(x){
  result <- run_microbiome_simulation_short(
    n_mice = n_mice,
    total_taxa = total_taxa,
    active_taxa = active_taxa,
    days_to_simulate = days_to_simulate,
    exchange_factor = x[1],
    network_density = x[2],
    extinction_threshold = extinction_threshold,
    presence_threshold = presence_threshold,
    structure = structure
  )
  
  adj_matrix <- result$adjacency_matrix
  graph <- cbind(result$microbiome_history[[days_to_simulate+1]], adj_matrix)
  return(graph)
}


# Make training, test and validation dataset and save it
run_sim <- function(n_train, n_val, n_test, attr, filepath = "case_study_mice/data/"){
  n <- c(n_train, n_val, n_test)
  name <- c("train", "val", "test")
  for(i in 1:3){
    print(paste0("Simulate ", name[i], " data"))
    vary_params <- prior(n = n[i])
    results <- lapply(1:nrow(vary_params), function(j){ 
      if(j %% 500 == 0){print(j)}
      simulator(as.numeric(vary_params[j,]))})
    array_results <- array(unlist(results), dim = c(n_mice, n_mice+total_taxa, length(results)))
    saveRDS(array_results, file = paste0(filepath, "simulation_output_array_", name[i], "_", attr ,".rds"))
    saveRDS(vary_params, file = paste0(filepath, "simulation_output_params_", name[i], "_", attr, ".rds"))
  }
}

for(i in 1:5){
  ### Run the simulation for horizon 5
  days_to_simulate <- 5
  run_sim(n_train = 50000, n_val = 1000, n_test = 1000, attr = paste0("simulator_day_5_n50k_",i))
  
  ### Run the simulation for horizon 10
  days_to_simulate <- 10
  run_sim(n_train = 50000, n_val = 1000, n_test = 1000, attr = paste0("simulator_day_10_n50k_",i))
  ### Run the simulation for horizon 30
  days_to_simulate <- 30
  run_sim(n_train = 50000, n_val = 1000, n_test = 1000, attr = paste0("simulator_day_30_n50k_",i))
}


