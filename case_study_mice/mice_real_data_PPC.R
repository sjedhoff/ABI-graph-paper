################################################################################
##                                                                            ##
##        Case Study Mice: Real Data : Posterior predictive checks            ##
##                                                                            ##
################################################################################

################################################################################
##                           Load posterior draws                             ##
################################################################################
library(R.matlab)
post_draws <- readMat("case_study_mice/real_data/social_graph_posteriors.mat")
post_draws <- post_draws$exchange.factor[1,,1]

################################################################################
##                Simulate taxa corresponding to posterior draws              ##
################################################################################
source("case_study_mice/mice_simulation_helper_functions.R")
adj_matrix <- readRDS("case_study_mice/real_data/data_ready/social_distances_matrix_sub.rds")

# fixed hyper-parameters
n_mice <- 177
total_taxa <- 52
active_taxa <- 5 
days_to_simulate <- 30
extinction_threshold <- 0.1 
presence_threshold <- 0.1

run_microbiome_simulation_short <- function(n_mice = 20, 
                                            total_taxa = 1000, 
                                            active_taxa = 300,
                                            days_to_simulate = 30,
                                            exchange_factor = 0.001,
                                            adj_matrix,
                                            extinction_threshold = 0.001,
                                            presence_threshold = 0.001) {
  
  
  
  
  # 1. generate network
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
      extinction_threshold = extinction_threshold,
      presence_threshold = presence_threshold
    ),
    adjacency_matrix    = adj_matrix,
    microbiome_history  = microbiome_history
  ))
}

data_posteriors <- lapply(post_draws, function(ex_factor) run_microbiome_simulation_short(n_mice, total_taxa,
                                                                               active_taxa, days_to_simulate,
                                                                               exchange_factor = ex_factor,
                                                                               adj_matrix, 
                                                                               extinction_threshold,
                                                                               presence_threshold))

saveRDS(data_posteriors, file = "case_study_mice/real_data/social_graph_posterior_graphs.rds")

data_microbiome <- lapply(data_posteriors, function(dat) dat$microbiome_history[[days_to_simulate + 1]])


################################################################################
##                Posterior predictive checks: Jaccard Index                  ##
################################################################################

jaccard_index_per_dataset <- function(data, presence_threshold = 0){
  n_mice <- nrow(data)
  sim_matrix <- matrix(0, nrow = n_mice, ncol = n_mice)
  
  # Pairwise Jaccard distances
  for(mouse1 in 1:n_mice){
    for(mouse2 in 1:n_mice){
      if(mouse1 == mouse2){
        sim_matrix[mouse1, mouse2] <- 1
      }
      else{
        presence1 <- data[mouse1, ] > presence_threshold
        presence2 <- data[mouse2, ] > presence_threshold
        
        intersection <- sum(presence1 & presence2)
        union <- sum(presence1 | presence2)
        
        jaccard_val <- ifelse(union > 0, intersection / union, 0)
        
        sim_matrix[mouse1, mouse2] <- jaccard_val
        sim_matrix[mouse2, mouse1] <- jaccard_val
        
        
      }
    }
  }
  
  mean_jaccard <- mean(sim_matrix[lower.tri(sim_matrix)])
  sd_jaccard <- sd(sim_matrix[lower.tri(sim_matrix)])
  
  return(list("mean_jaccard" = mean_jaccard,
              "sd_jaccard" = sd_jaccard))
}

jaccard_values_posteriors <- lapply(data_microbiome, jaccard_index_per_dataset)


## for the true dataset
social_graph <- readRDS("case_study_mice/real_data/data_ready/social_graph_selected.rds")
true_microbiome <- social_graph[,1:52]
jaccard_values_true <- jaccard_index_per_dataset(true_microbiome)


## Plots

# one dimensional
mean_jaccards <- unlist(lapply(jaccard_values_posteriors, function(x) x[[1]]))
hist(mean_jaccards, main = "Mean of Jaccard Distances", xlim = c(0.1,0.45))
abline(v = jaccard_values_true$mean_jaccard, col = "red", lwd = 3)

sd_jaccards <- unlist(lapply(jaccard_values_posteriors, function(x) x[[2]]))
hist(sd_jaccards, main = "Standard deviation of Jaccard Distances")
abline(v = jaccard_values_true$sd_jaccard, col = "red", lwd = 3)



## Compare to prior draws:
exchange_factor_prior <- seq(from = 0.05, to = 0.95, length.out = 100)

data_priors <- lapply(exchange_factor_prior, function(ex_factor) run_microbiome_simulation_short(n_mice, total_taxa,
                                                                                          active_taxa, days_to_simulate,
                                                                                          exchange_factor = ex_factor,
                                                                                          adj_matrix, 
                                                                                          extinction_threshold,
                                                                                          presence_threshold))
data_microbiome_prior <- lapply(data_priors, function(dat) dat$microbiome_history[[days_to_simulate + 1]])
jaccard_values_priors <- lapply(data_microbiome_prior, jaccard_index_per_dataset)

mean_jaccards_prior <- unlist(lapply(jaccard_values_priors, function(x) x[[1]]))
sd_jaccards_prior <- unlist(lapply(jaccard_values_priors, function(x) x[[2]]))

plot_df <- data.frame("type" = "Real-world dataset",
           "mean" = jaccard_values_true[[1]],
           "sd" = jaccard_values_true[[2]])
plot_df <- rbind(plot_df, data.frame("type" = rep("Prior draw", 100),
                                     "mean" = mean_jaccards_prior,
                                     "sd" = sd_jaccards_prior))
plot_df <- rbind(plot_df, data.frame("type" = rep("Posterior draw", 500),
                                     "mean" = mean_jaccards,
                                     "sd" = sd_jaccards))

base_cols <- c("#1f77b4", "#d62728", "#2ca02c", "#9467bd")

ggplot(plot_df, aes(x = mean, y = sd, col = factor(type), alpha = factor(type))) +
  geom_point(size = 2) +
  labs(x = "Mean Jaccard index", y = "Standard Deviation \n of Jaccard index") +
  scale_color_manual(values = base_cols[c(2,1,3)], name = "") +
  scale_alpha_manual(values = c(0.5,0.8,1), name = "")+
  theme_light() +
  theme(
    strip.background.y = element_blank(),
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 8),
    legend.box.margin = margin(t = -6, r = 0, b = 0, l = 0),
    legend.margin = margin(0, 0, 0, 0),
    legend.position = "bottom"
  )
ggsave(filename = "plots/mice_real_data_jaccard.pdf",
       units = "in", width = 6.8, height = 2.5)





