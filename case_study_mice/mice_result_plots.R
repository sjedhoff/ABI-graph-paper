################################################################################
##                                                                            ##
##                     Case Study Mice: Plotting results                      ##
##                                                                            ##
################################################################################

library(tidyr)
library(dplyr)
library(ggplot2)
library(xtable)

## Load the data
metrics <- read.csv("case_study_mice/results/all_metrics.csv", header=FALSE)
val_loss <- read.csv("case_study_mice/results/val_losses.csv")


colnames(metrics) <- c("day", "run", "summary_network", "PC_ND","PC_EF","LG_ND","LG_EF","R_ND","R_EF","VAL_LOSS_LAST")


## Bring the data in the right format
val_long <- val_loss %>%
  tidyr::pivot_longer(
    cols = -c(day, run),
    names_to = "abbr",
    values_to = "VAL_LOSS_LAST"
  ) %>%
  mutate(
    # map val_loss columns -> summary_network labels
    summary_network = dplyr::recode(
      abbr,
      gcn = "GCN",
      ds  = "DeepSet",
      st  = "SetTransformer",
      gt  = "GraphTransformer"
    ),
    # turn NaN into proper NA
    VAL_LOSS_LAST = ifelse(is.nan(VAL_LOSS_LAST), NA_real_, VAL_LOSS_LAST)
  ) %>%
  select(day, run, summary_network, VAL_LOSS_LAST)

metrics <- metrics %>%
  select(-VAL_LOSS_LAST) %>%                   # drop old column if present
  left_join(val_long, by = c("day", "run", "summary_network"))





### ---- Table for the Appendix ---- ##

metrics_out <- metrics %>%
  group_by(day, summary_network) %>%
  summarise("val_loss_median" = median(VAL_LOSS_LAST),
            "R_ND_median" = median(R_ND),
            "R_EF_median" = median(R_EF),
            "LG_ND_median" = median(LG_ND),
            "LG_EF_median" = median(LG_EF),
            "PC_ND_median" = median(PC_ND),
            "PC_EF_median" = median(PC_EF))
print(xtable(metrics_out, digits = 2), include.rownames = FALSE)


### ---- Plots ---- ##

df <- metrics %>%
  pivot_longer(
    cols = c("PC_ND", "PC_EF", "LG_ND", "LG_EF", "R_ND", "R_EF"),
    names_to = c("type", "parameter"),
    names_pattern = "^(R|PC|LG)_(ND|EF)",
    values_to = "value"
  ) %>%
  mutate(
    summary_networks = factor(summary_network),
    type = factor(type, levels = c("R", "PC", "LG"), labels = c("Recovery", "Posterior Contraction", "Calibration")),
    parameter = factor(parameter, levels = c("ND", "EF"), labels = c("Network Density", "Exchange Factor")),
    day = factor(day)) %>%
  select(value, parameter, summary_networks, day, type) %>%
  mutate(
    value = case_when(
      !is.finite(value) ~ -100,
      TRUE ~ value
    )
  ) 



base_cols <- c("#1f77b4", "#d62728", "#2ca02c", "#9467bd")
df_pi <- df %>%  
  mutate(
    summary_networks  = factor(summary_networks)
  )

sum_df <- df_pi %>%
  group_by(type, summary_networks, day, parameter) %>%
  summarise(
    med = median(value, na.rm = TRUE),
    lo = min(value),
    hi = max(value),
    .groups = "drop"
  )

pd <- position_dodge(width = 0.6)
x_labs <- c(
  "DeepSet" = "DS",
  "GCN" = "GCN",
  "GraphTransformer" = "GT",
  "SetTransformer" = "ST"
)

p0 <- sum_df %>%
  filter(type == "Recovery") %>%
  ggplot(aes(
    x = summary_networks, y = med,
    color = summary_networks,
    shape = day
  )) +
  geom_linerange(aes(ymin = lo, ymax = hi), position = pd, linewidth = 0.5) +
  geom_point(position = pd, stroke = 0.8) +
  facet_grid(parameter ~ type) +
  scale_color_manual(values = base_cols, name = "Summary network") +
  scale_shape( name = "Day") +
  scale_x_discrete(labels = x_labs) +
  guides(
    color = "none",
    shape = "none"
  ) +
  labs(x = NULL, y = NULL) +
  scale_y_continuous(guide = "axis", limits = c(0,1)) +
  theme_light() +
  theme(
    strip.text.y = element_blank(),   # remove row facet labels
    strip.background.y = element_blank(),
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 8),
    legend.box.margin = margin(t = -6, r = 0, b = 0, l = 0),
    legend.margin = margin(0, 0, 0, 0),
  )
p0

p1 <- sum_df %>%
  filter(type == "Posterior Contraction") %>%
  ggplot(aes(
    x = summary_networks, y = med,
    color = summary_networks,
    shape = day
  )) +
  geom_linerange(aes(ymin = lo, ymax = hi), position = pd, linewidth = 0.5) +
  geom_point(position = pd, stroke = 0.8) +
  facet_grid(parameter ~ type) +
  scale_color_manual(values = base_cols, name = "Summary network") +
  scale_shape( name = "Day") +
  scale_x_discrete(labels = x_labs) +
  guides(
    color = "none",
    shape = "none"
  ) +
  labs(x = NULL, y = NULL) +
  scale_y_continuous(guide = "axis", limits = c(0,1)) +
  theme_light() +
  theme(
    strip.text.y = element_blank(),   # remove row facet labels
    strip.background.y = element_blank(),
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 8),
    legend.box.margin = margin(t = -6, r = 0, b = 0, l = 0),
    legend.margin = margin(0, 0, 0, 0),
  )
p1


p2 <- sum_df %>%
  filter(type == "Calibration") %>%
  ggplot(aes(
    x = summary_networks, y = med,
    color = summary_networks,
    shape = day
  )) +
  geom_linerange(aes(ymin = lo, ymax = hi), position = pd, linewidth = 0.5) +
  geom_point(position = pd, stroke = 0.8) +
  facet_grid(parameter ~ type, scales = "free_y") + #, labeller = label_parsed) +
  scale_color_manual(values = base_cols, name = "Summary network",
                     labels = c("DeepSets", "GCN", "GraphTransformer", "SetTransformer")) +
  scale_shape(name = "Observation horizon", labels = c("5 days", "10 days", "30 days")) +
  scale_x_discrete(labels = x_labs) +
  guides(
    color = guide_legend(ncol = 1, order = 1),
    shape = guide_legend(ncol = 1)
  ) +
  labs(x = NULL, y = NULL) +
  theme_light() +
  theme(
    #strip.text.y = element_text(angle = 0),
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 8),
    legend.box.margin = margin(t = -6, r = 0, b = 0, l = 0),
    legend.margin = margin(0, 0, 0, 0),
  )

library(patchwork)
p <- p0+p1+p2 + plot_layout(widths = c(1, 1, 0.95))
p

ggsave(p, filename = "plots/mice_results.pdf",
       units = "in", width = 6.8, height = 3)




