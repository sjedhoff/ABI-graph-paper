################################################################################
##                                                                            ##
##                Toy Example: Analytical likelihood - MCMC                   ##
##                                                                            ##
################################################################################
library(bayesplot)
library(cmdstanr)
library(jsonlite) 
library(dplyr)
library(SBC)


# Prior and Simulator
################################################################################
N_NODES <- 30

prior <- function() {
  list(
    pi_aa = runif(1, min = 0.1, max = 0.9),
    pi_bb = runif(1, min = 0.1, max = 0.9),
    pi_ab = runif(1, min = 0.1, max = 0.9),
    num_a = sample(5:25, 1)
  )
}


simulator <- function(pi_aa, pi_bb, pi_ab, num_a, n_nodes = N_NODES) {
  adj <- matrix(0L, nrow = n_nodes, ncol = n_nodes)
  x   <- integer(n_nodes)
  x[1:num_a] <- 1L
  
  for (i in 1:(n_nodes - 1)) {
    for (j in (i + 1):n_nodes) {
      if (x[i] == x[j]) {
        p_edge <- ifelse(x[i] == 1, pi_aa, pi_bb)
      } else {
        p_edge <- pi_ab
      }
      edge       <- rbinom(1, size = 1, prob = p_edge)
      adj[i, j]  <- edge
      adj[j, i]  <- edge
    }
  }
  
  list(x = x, adj = adj)
}

# Stan Model
################################################################################

model <- cmdstan_model("case_study_toy/analytical_likelihood_case/stan/analytical_likelihood.stan")


# Draw from prior and simulate
theta <- prior()
obs   <- simulator(
  pi_aa = theta$pi_aa,
  pi_bb = theta$pi_bb,
  pi_ab = theta$pi_ab,
  num_a = theta$num_a
)

# Package data for Stan
stan_data <- list(
  N      = N_NODES,
  x      = obs$x,
  adj    = obs$adj
)

# Fit
fit <- model$sample(
  data         = stan_data,
  iter_warmup  = 1000,
  iter_sampling = 1000,
  chains       = 4,
  parallel_chains = 4,
  seed         = 42
)

fit$summary(variables = c("pi_aa", "pi_bb", "pi_ab"))



mcmc_hist(fit$draws("pi_aa"))
mcmc_hist(fit$draws("pi_bb"))
mcmc_hist(fit$draws("pi_ab"))


# Running SBC
################################################################################

dataset_generator <- function() {
  theta <- prior()
  g     <- do.call(simulator, theta)
  
  list(
    variables = list(
      pi_aa = theta$pi_aa,
      pi_bb = theta$pi_bb,
      pi_ab = theta$pi_ab
    ),
    generated = list(
      N      = N_NODES,
      x      = g$x,
      adj    = g$adj
    )
  )
}


set.seed(42)
generator <- SBC_generator_function(dataset_generator)
datasets  <- generate_datasets(generator, n_sims = 500)

## save the datasets to load them into python as well
datasets_to_save <- lapply(seq_len(length(datasets$generated)), function(i) {
  list(
    # true parameters
    pi_aa = datasets$variables[i,1],
    pi_bb = datasets$variables[i,2],
    pi_ab = datasets$variables[i,3],
    # observed data
    x     = datasets$generated[[i]]$x,
    adj   = datasets$generated[[i]]$adj
  )
})

saveRDS(datasets, file = "case_study_toy/analytical_likelihood_case/results/sbc_datasets.rds")
write_json(datasets_to_save, "case_study_toy/analytical_likelihood_case/results/sbc_datasets.json",
           digits = 10, auto_unbox = TRUE)

## run SBC
backend <- SBC_backend_cmdstan_sample(
  model,
  iter_warmup     = 1000,
  iter_sampling   = 125,
  chains          = 4,
  parallel_chains = 4
)

datasets <- readRDS("case_study_toy/analytical_likelihood_case/results/sbc_datasets.rds")

ll <- function(x, adj, pi_aa, pi_bb, pi_ab, N){
  ll <- 0
  base_p <- matrix(NA, nrow = N, ncol = N)
  for (i in 1:(N - 1)) {
    for (j in (i + 1):N) {
      if (x[i] == x[j]) {
        base_p[i,j] <- ifelse(x[i] == 1, pi_aa, pi_bb)
      } else {
        base_p[i,j] <- pi_ab
      }
      ll <- ll + dbinom(adj[i,j], size = 1, p = base_p[i,j], log = TRUE)
    }
  }
  return(ll)
}
dq <- derived_quantities(log_lik = ll(x, adj, pi_aa, pi_bb, 
                                      pi_ab, N = N_NODES),
                         .globals = "ll")

sbc_time <- system.time(results <- compute_SBC(datasets = datasets, backend = backend,
                                               dquants = dq))

saveRDS(results, file = "case_study_toy/analytical_likelihood_case/results/sbc_mcmc_results.rds")

# SBC Diagnostics
################################################################################
results <- readRDS("case_study_toy/analytical_likelihood_case/results/sbc_mcmc_results.rds")

## Calibration plots
x <- results
ecdf_data <- SBC:::data_for_ecdf_plots(x, prob = 0.95)
N <- ecdf_data$N
K <- ecdf_data$K
z <- ecdf_data$z

ecdf_df <- dplyr::mutate(ecdf_data$ecdf_df, z_diff = ecdf - z, type = "sample ECDF")
limits_df_trans <- dplyr::mutate(ecdf_data$limits_df,
                                 ymax = upper - uniform_val,
                                 ymin = lower - uniform_val,
                                 type = "theoretical CDF"
)
facet_args <- list()
size <- 1
all_facet_args <- c(list(~group, scales = "free_y"), facet_args)
all_facet_args$nrow <- 1
all_facet_args$labeller <- as_labeller(c(
  "log_lik" = "Log-Likelihood",
  "pi_aa" = expression("pi[AA]"),
  "pi_bb" = expression("pi[BB]"),
  "pi_ab" = expression("pi[AB]")
), label_parsed)

ecdf_df %>%
  mutate(group = factor(group, levels = c("log_lik", "pi_aa", "pi_bb", "pi_ab"),
                           labels = c("Log-Likelihood", expression("pi[AA]"), expression("pi[BB]"), expression("pi[AB]")))) %>%
ggplot(aes(color = type, fill = type)) +
  geom_ribbon(
    data = limits_df_trans,
    aes(x = x, ymax = ymax, ymin = ymin),
    alpha = 0.33,
    linewidth = size, color = NA) +
  geom_step(
    aes(x = z, y = z_diff, group = variable, alpha = alpha),
    lwd = 1
  ) +
  scale_color_manual(
    name = "",
    values = rlang::set_names(
      c("grey", "#132a70"),
      c("theoretical CDF", "sample ECDF")),
    labels = c(
      "theoretical CDF" = expression(italic("theoretical CDF")),
      "sample ECDF" = expression(italic("sample ECDF"))
    )
  ) +
  scale_fill_manual(
    name = "",
    values = c("theoretical CDF" = "grey",
               "sample ECDF" = "transparent"),
    labels = c(
      "theoretical CDF" = expression(italic("theoretical CDF")),
      "sample ECDF" = expression(italic("sample ECDF"))
    )
  ) +
  scale_alpha_identity() +
  xlab("Fractional rank statistic") +
  ylab("ECDF Difference") +
  theme_minimal() +
  facet_wrap(~group, scales = "free_y", nrow = 1, 
             labeller = label_parsed) +
  theme(legend.position = "none",
        axis.line = element_line(color = "black"),
        axis.text  = element_text(size = 12),  # tick labels
        axis.title = element_text(size = 16),  # axis titles
        strip.text = element_text(size = 18),   # facet headers
        plot.title = element_text(size = 30, hjust = 0.5)
        ) +
  ggtitle("MCMC")
#ggsave("plots\\toy_analytical_likelihood_ecdf_mcmc.pdf", height = 5, width = 20, units = "in")





## Save median and credible intervals
mcmc_stats <- results$stats |>
  filter(variable %in% c("pi_aa", "pi_bb", "pi_ab")) |>
  select(variable, median, q5, q95)  # or q2.5, q97.5 for 95% CI

write_json(
  list(
    median = as.list(setNames(
      lapply(c("pi_aa","pi_bb","pi_ab"), function(v) mcmc_stats$median[mcmc_stats$variable == v]),
      c("pi_aa","pi_bb","pi_ab"))),
    q5   = as.list(setNames(
      lapply(c("pi_aa","pi_bb","pi_ab"), function(v) mcmc_stats$q5[mcmc_stats$variable == v]),
      c("pi_aa","pi_bb","pi_ab"))),
    q95  = as.list(setNames(
      lapply(c("pi_aa","pi_bb","pi_ab"), function(v) mcmc_stats$q95[mcmc_stats$variable == v]),
      c("pi_aa","pi_bb","pi_ab")))
  ),
  "case_study_toy/analytical_likelihood_case/results/mcmc_posterior_stats.json", digits = 10, auto_unbox = TRUE
)


