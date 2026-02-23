library(dplyr)
library(tidyr)
library(ggplot2)
library(patchwork)


# Load the data
################################################################################

metrics <- read.csv("case_study_toy/results/all_metrics.csv")
loss <- read.csv("case_study_toy/results/val_losses.csv")

networks <- strsplit(metrics$workflow, "-")
metrics$summary_networks <- unlist(lapply(networks, function(x) x[1]))
metrics$aggregation_layer <- unlist(lapply(networks, function(x) x[2]))

map <- c(
  "SetTrans"   = "SetTransformer",
  "GraphTrans" = "GraphTransformer",
  "DeepSet"    = "DeepSets"
)
metrics <- metrics %>% mutate(summary_networks = recode(summary_networks, !!!map))

metrics <- metrics %>%                
  left_join(loss, by = c("run", "workflow"))

# Plots
################################################################################
metrics_long_all <- metrics %>%
  pivot_longer(
    cols = matches("^(R|PC|LG)_(pi_aa|pi_ab|pi_bb|gamma)$"),
    names_to = c("type", "variable"),
    names_pattern = "^(R|PC|LG)_(pi_aa|pi_ab|pi_bb|gamma)$",
    values_to = "value"
  ) %>%
  mutate(
    summary_networks = factor(summary_networks),
    aggregation_layer = factor(aggregation_layer,
                               levels = c("MeanPooling", "InvariantLayer", "MHAttention"),
                               labels = c("Mean", "Invariant", "PMA")),
    type = factor(type, levels = c("R", "PC", "LG"), labels = c("Recovery", "Posterior Contraction", "Calibration")),   # columns order in grid
    metric = if_else(variable %in% c("pi_aa","pi_ab","pi_bb"), "pi", variable),
    metric = factor(metric, levels = c("pi", "gamma"), labels = c(expression("pi"), expression("lambda")))
  ) %>%
  select(metric, value, summary_networks, aggregation_layer, type) %>%
  mutate(
    value = case_when(
      !is.finite(value) ~ -50,
      TRUE ~ value
    )
  )

df_plot <- metrics_long_all %>%
  group_by(type, metric, summary_networks, aggregation_layer) %>%
  summarise(
    med = median(value, na.rm = TRUE),
    lo  = min(value, na.rm = TRUE),
    hi  = max(value, na.rm = TRUE),
    .groups = "drop"
  )

base_cols <- c("#1f77b4", "#d62728", "#2ca02c", "#9467bd")
pd <- position_dodge(width = 0.6)
x_labs <- c(
  "DeepSets" = "DS",
  "GCN" = "GCN",
  "GraphTransformer" = "GT",
  "SetTransformer" = "ST"
)

df_plot %>%
  filter() %>%
  ggplot(aes(x = summary_networks, y = med,
             color = summary_networks,
             shape = aggregation_layer)) +
  geom_linerange(aes(ymin = lo, ymax = hi), linewidth = 1, position = pd) +
  geom_point(size = 2.6, stroke = 0.8, position = pd) +
  facet_wrap(metric ~ type, scales = "free_y") +
  scale_color_manual(values = base_cols, name = "Summary network") +
  scale_shape( name = "Aggregation layer") +
  scale_x_discrete(labels = x_labs)

### Plots for the paper
p1 <- df_plot %>%
  filter(type == "Recovery") %>%
  ggplot(aes(x = summary_networks, y = med,
             color = summary_networks,
             shape = aggregation_layer)) +
  geom_linerange(aes(ymin = lo, ymax = hi), linewidth = 0.5, position = pd) +
  geom_point(position = pd) +
  facet_grid(metric ~ type, scales = "free_y") +
  scale_color_manual(values = base_cols, name = "Summary network") +
  scale_shape( name = "Aggregation layer") +
  scale_x_discrete(labels = x_labs) +
  guides(color = "none", shape = "none") +
  scale_y_continuous(guide = "axis", limits = c(0,1), name = "") +
  labs(x = "") +
  theme_light() +
  theme(strip.text.y = element_blank(),
        plot.margin = margin(0, 0, 0, 0))
p1

p2 <- df_plot %>%
  filter(type == "Posterior Contraction") %>%
  ggplot(aes(x = summary_networks, y = med,
             color = summary_networks,
             shape = aggregation_layer)) +
  geom_linerange(aes(ymin = lo, ymax = hi), linewidth = 0.5, position = pd) +
  geom_point(position = pd) +
  facet_grid(metric ~ type, scales = "free_y") +
  scale_color_manual(values = base_cols, name = "Summary network") +
  scale_shape( name = "Aggregation layer") +
  scale_x_discrete(labels = x_labs) +
  guides(color = "none", shape = "none") +
  scale_y_continuous(guide = "axis", limits = c(0,1), name = "") +
  labs(x = "") +
  theme_light() +
  theme(strip.text.y = element_blank(),
        plot.margin = margin(0, 0, 0, 0))
p2

p3 <- df_plot %>%
  filter(type == "Calibration") %>%
  ggplot(aes(x = summary_networks, y = med,
             color = summary_networks,
             shape = aggregation_layer)) +
  geom_linerange(aes(ymin = lo, ymax = hi), linewidth = 0.5, position = pd) +
  geom_point(position = pd) +
  facet_grid(metric ~ type, scales = "free_y", labeller = label_parsed) +
  scale_color_manual(values = base_cols, name = "Summary network") +
  scale_shape( name = "Aggregation layer") +
  scale_x_discrete(labels = x_labs) +
  guides(color  = guide_legend(order = 1)) +
  labs(x = "", y = "") +
  theme_light() +
  theme(legend.title = element_text(size = 10),
        legend.text = element_text(size = 8),
        legend.box.margin = margin(t = -6, r = 0, b = 0, l = 0),
        legend.margin = margin(0, 0, 0, 0),
        plot.margin = margin(0, 0, 0, 0),
        strip.text.y = element_text(angle = 0))
p3
p <- p1+p2+p3 + plot_layout(widths = c(1, 1, 0.95)) 
p
ggsave(p, filename = "plots/toy_example_comparison.pdf",
       units = "in", width = 6.8, height = 3)



# only one aggregation layer each
################################################################################
df_plot_small <- df_plot %>%
  filter((summary_networks == "GCN" & aggregation_layer == "Mean") |
           (summary_networks == "DeepSets" & aggregation_layer == "Invariant") |
           (summary_networks == "GraphTransformer" & aggregation_layer == "PMA") |
           (summary_networks == "SetTransformer" & aggregation_layer == "PMA"))


p1 <- df_plot_small %>%
  filter(type == "Recovery") %>%
  ggplot(aes(x = summary_networks, y = med,
             color = summary_networks)) +
  geom_linerange(aes(ymin = lo, ymax = hi), linewidth = 0.5, position = pd) +
  geom_point(position = pd) +
  facet_grid(metric ~ type, scales = "free_y") +
  scale_color_manual(values = base_cols, name = "Summary network") +
  scale_x_discrete(labels = x_labs) +
  guides(color = "none", shape = "none") +
  scale_y_continuous(guide = "axis", limits = c(0,1), name = "") +
  labs(x = "") +
  theme_light() +
  theme(strip.text.y = element_blank(),
        plot.margin = margin(0, 0, 0, 0))
p1

p2 <- df_plot_small %>%
  filter(type == "Posterior Contraction") %>%
  ggplot(aes(x = summary_networks, y = med,
             color = summary_networks)) +
  geom_linerange(aes(ymin = lo, ymax = hi), linewidth = 0.5, position = pd) +
  geom_point(position = pd) +
  facet_grid(metric ~ type, scales = "free_y") +
  scale_color_manual(values = base_cols, name = "Summary network") +
  scale_x_discrete(labels = x_labs) +
  guides(color = "none", shape = "none") +
  scale_y_continuous(guide = "axis", limits = c(0,1), name = "") +
  labs(x = "") +
  theme_light() +
  theme(strip.text.y = element_blank(),
        plot.margin = margin(0, 0, 0, 0))
p2

p3 <- df_plot_small %>%
  filter(type == "Calibration") %>%
  ggplot(aes(x = summary_networks, y = med,
             color = summary_networks)) +
  geom_hline(yintercept = 0, col = "grey") +
  geom_linerange(aes(ymin = lo, ymax = hi), linewidth = 0.5, position = pd) +
  geom_point(position = pd) +
  facet_grid(metric ~ type, scales = "free_y", labeller = label_parsed) +
  scale_color_manual(values = base_cols, name = "Summary network") +
  scale_x_discrete(labels = x_labs) +

  guides(color  = guide_legend(order = 1)) +
  labs(x = "", y = "") +
  theme_light() +
  theme(legend.title = element_text(size = 10),
        legend.text = element_text(size = 8),
        legend.box.margin = margin(t = -6, r = 0, b = 0, l = 0),
        legend.margin = margin(0, 0, 0, 0),
        plot.margin = margin(0, 0, 0, 0),
        strip.text.y = element_text(angle = 0))
p3
p <- p1+p2+p3 + plot_layout(widths = c(1, 1, 0.95)) 
p
ggsave(p, filename = "plots/toy_example_comparison_presentation.png",
       units = "in", width = 6.8, height = 3)




# Table
################################################################################

model_order  <- c("DeepSets", "GCN","GraphTransformer", "SetTransformer")
pool_order   <- c("MeanPooling", "InvariantLayer", "MHAttention")


metrics_out <- metrics %>%
  group_by(summary_networks, aggregation_layer) %>%
  summarise(
    "number_pars" = median(number_parameters),
    "Loss" = median(last_loss),
    "PR_pi" = median(c(R_pi_aa, R_pi_bb, R_pi_ab)),
    "PR_gamma" = median(R_gamma),
    "LG_pi" = median(c(LG_pi_aa, LG_pi_bb, LG_pi_ab)),
    "LG_gamma" = median(LG_gamma),
    "PC_pi" =  median(c(PC_pi_aa, PC_pi_bb, PC_pi_ab)),
    "PC_gamma" = median(PC_gamma),
  )


metrics_out <- metrics_out %>%
  mutate(
    summary_networks = factor(summary_networks, levels = model_order),
    aggregation_layer  = factor(aggregation_layer,  levels = pool_order)
  ) %>%
  arrange(summary_networks, aggregation_layer)


library(xtable)
print(xtable(metrics_out, digits = 2), include.rownames = FALSE)


# Calibration of data-dependent test quantities
################################################################################
SBC_data <- read_csv("case_study_toy/results/SBC_data.csv")

networks <- strsplit(SBC_data$wf_name, "-")
SBC_data$summary_networks <- unlist(lapply(networks, function(x) x[1]))
SBC_data$aggregation_layer <- unlist(lapply(networks, function(x) x[2]))

map <- c(
  "SetTrans"   = "SetTransformer",
  "GraphTrans" = "GraphTransformer",
  "DeepSet"    = "DeepSets"
)
SBC_data <- SBC_data %>% mutate(summary_networks = recode(summary_networks, !!!map))

map <- c(
  "spectral_gap" = "Spectral Gap",
  "edge_density" = "Edge Density",
  "degree_assortativity" = "Degree Assort.",
  "global_clustering" = "Global Clustering"
)
SBC_data <- SBC_data %>% mutate(metric = recode(metric, !!!map))

SBC_data <- SBC_data %>%
  mutate(metric = factor(metric, levels = c("Spectral Gap","Edge Density",
                                              "Degree Assort.","Global Clustering")))

df_plot <- SBC_data %>%
  group_by(metric, summary_networks, aggregation_layer) %>%
  summarise(
    med_lg = median(lg_val, na.rm = TRUE),
    lo_lg  = min(lg_val,    na.rm = TRUE),
    hi_lg  = max(lg_val,    na.rm = TRUE),
    med_r  = median(r_val,  na.rm = TRUE),
    lo_r   = min(r_val,     na.rm = TRUE),
    hi_r   = max(r_val,     na.rm = TRUE),
    .groups = "drop"
  )%>%
  pivot_longer(
    cols      = c(med_lg, lo_lg, hi_lg, med_r, lo_r, hi_r),
    names_to  = c(".value", "value_type"),
    names_pattern = "(med|lo|hi)_(lg|r)"
  ) %>%
  mutate(value_type = recode(value_type, "lg" = "Calibration", "r" = "Recovery")) %>%
  mutate(metric = factor(metric, levels =   c("Spectral Gap","Edge Density",
                          "Degree Assort.","Global Clustering")))

base_cols <- c("#1f77b4", "#d62728", "#2ca02c", "#9467bd")
pd <- position_dodge(width = 0.6)
x_labs <- c(
  "DeepSets" = "DS",
  "GCN" = "GCN",
  "GraphTransformer" = "GT",
  "SetTransformer" = "ST"
)


p1 <- df_plot %>%
  filter(value_type == "Recovery") %>%
  ggplot(aes(x = summary_networks, y = med,
             color = summary_networks,
             shape = aggregation_layer)) +
  geom_linerange(aes(ymin = lo, ymax = hi), linewidth = 0.5, position = pd) +
  geom_point(position = pd) +
  facet_grid(metric ~ value_type, scales = "free_y") +
  scale_color_manual(values = base_cols, name = "Summary network") +
  scale_shape( name = "Aggregation layer") +
  scale_x_discrete(labels = x_labs) +
  guides(color = "none", shape = "none") +
  scale_y_continuous(guide = "axis", limits = c(0,1), name = "") +
  labs(x = "") +
  theme_light() +
  theme(strip.text.y = element_blank(),
        plot.margin = margin(0, 0, 0, 0))
p1

p2 <- df_plot %>%
  filter(value_type == "Calibration") %>%
  ggplot(aes(x = summary_networks, y = med,
             color = summary_networks,
             shape = aggregation_layer)) +
  geom_linerange(aes(ymin = lo, ymax = hi), linewidth = 0.5, position = pd) +
  geom_point(position = pd) +
  facet_grid(metric ~ value_type, scales = "free_y") + 
  scale_color_manual(values = base_cols, name = "Summary network") +
  scale_shape( name = "Aggregation layer") +
  scale_x_discrete(labels = x_labs) +
  guides(color  = guide_legend(order = 1)) +
  labs(x = "", y = "") +
  theme_light() +
  theme(legend.title = element_text(size = 10),
        legend.text = element_text(size = 8),
        legend.box.margin = margin(t = -6, r = 0, b = 0, l = 0),
        legend.margin = margin(0, 0, 0, 0),
        plot.margin = margin(0, 0, 0, 0))
p2
p <- p1+p2 + plot_layout(widths = c(1, 0.95)) 
p
ggsave(p, filename = "plots/toy_example_SBC_data.pdf",
       units = "in", width = 6.8, height = 5)



### Table
model_order  <- c("DeepSets", "GCN","GraphTransformer", "SetTransformer")
pool_order   <- c("MeanPooling", "InvariantLayer", "MHAttention")


df_wide <- SBC_data %>%
  group_by(summary_networks, aggregation_layer, metric) %>%
  summarise(
    r_val  = median(r_val,  na.rm = TRUE),
    lg_val = median(lg_val, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_wider(
    names_from  = metric,
    values_from = c(r_val, lg_val),
    names_glue  = "{.value}_{metric}"
  ) %>%
  rename_with(~ stringr::str_replace(., "r_val_",  "Recovery_"),  starts_with("r_val_")) %>%
  rename_with(~ stringr::str_replace(., "lg_val_", "Calibration_"), starts_with("lg_val_"))


metrics_out <- df_wide %>%
  mutate(
    summary_networks = factor(summary_networks, levels = model_order),
    aggregation_layer  = factor(aggregation_layer,  levels = pool_order)
  ) %>%
  arrange(summary_networks, aggregation_layer)


library(xtable)
print(xtable(metrics_out, digits = 2), include.rownames = FALSE)

