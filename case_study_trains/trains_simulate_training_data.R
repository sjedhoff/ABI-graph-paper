################################################################################
##                                                                            ##
##         Case Study Trains: Generate Training Data for Bayesflow            ##
##                                                                            ##
################################################################################

source("case_study_trains/trains_simulator_function.R")
source("case_study_trains/trains_schedule_simulator_function.R")

### Fixed Adjacency Matrix
set.seed(31415)
adj <- adjacency_simulator(n_sections = 10)

### Function that calls the simulator:
# - N: number of different schedules run on the simulator
# - each: how often should each schedule be run
# - data_attr: string indicating how the datasets are saved
# - adj: adjacency matrix
run_simulator <- function(N = 1000, each = 128, data_attr, adj){
  # simulate N different schedules
  schedules <- replicate(N, schedule_simulator(adj = adj, n_trains = 4, n_stops = 4, overlaps = TRUE))
  schedules <- abind::abind(replicate(each, schedules, simplify = FALSE), along = 3)
  
  data <- lapply(1:(N*each), function(i){
    if(i %% 200 == 0){
      print(i)
    }
    # for each run: get new random default times
    sect_stats <- data.frame("default_time" = abs(round(runif(10, min = 5, max = 25))),
                             "prob_zero_delay" = rep(0.9, 10), # 10 percent random delay
                             "delay_lambda" = rep(0.5, 10)) 
    erg <- train_simulator(adj = adj, schedule = schedules[,,i], sect_stats)
  })
  inference_vars <- do.call(cbind,lapply(data, function(x) x[[1]]))
  graph <- do.call(abind::abind, list(lapply(data, function(x) x[[2]]), along = 3))
  dimnames(graph) <-  NULL
  # save the results accordingly
  saveRDS(inference_vars, file = paste0("case_study_trains/data/inference_vars_", data_attr, ".rds"))
  saveRDS(graph, file = paste0("case_study_trains/data/graph_", data_attr, ".rds"))
  
  default_times <- graph[,1,]
  return(list(schedules, default_times))
}

### Run the simulation and save it
data_attr <- "four_trains_sim_with_random_delay_large_testtest"
training_data <- run_simulator(N = 10000, each = 64, data_attr = data_attr, adj = adj)
val_data <- run_simulator(N = 200, each = 64, data_attr = paste0(data_attr, "_val"), adj = adj)
test_data <- run_simulator(N = 100, each = 1, data_attr = paste0(data_attr, "_test"), adj = adj)


### Calculating the ground truth distributions (via samples) for the test data
make_ground_truth_ <- function(data_attr, adj, schedules, default_times, n_sample = 500){
  N <- dim(schedules)[3]
  ma <- array(NA, dim = c(N, n_sample, 4))
  for(i in 1:N){
    sch <- schedules[,,i]
    sect_stats <- data.frame("default_time" = default_times[,i],
                             "prob_zero_delay" = rep(0.9, 10), ## 10 percent delay 
                             "delay_lambda" = rep(0.5, 10)) 
    samples <- lapply(1:n_sample, function(j) train_simulator(adj = adj, schedule = sch, section_stats = sect_stats)[[1]])
    m <- do.call(rbind, samples)
    ma[i,,] <- m
    if(i %% 10 == 0){
      print(i)
    }
  }
  
  saveRDS(ma, file = paste0("case_study_trains/data/ground_truth_", data_attr, ".rds"))
  return(ma)
}

gt <- make_ground_truth_(paste0(data_attr, "_test"), adj, schedules = test_data[[1]], 
                         default_times = test_data[[2]])
make_ground_truth_(paste0("2_", data_attr, "_test"), adj, schedules = test_data[[1]], 
                   default_times = test_data[[2]])
