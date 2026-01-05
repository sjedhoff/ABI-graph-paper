################################################################################
##                                                                            ##
##  Case Study Mice: Base functions for simulating the interaction network    ##
##                                                                            ##
################################################################################


library(igraph)
library(ggplot2)
library(reshape2)
library(RColorBrewer)
library(dplyr)
library(tidyr)
library(gridExtra)

# force printing of gtable objects (like those from arrangeGrob)
print.gtable <- function(x, ...) {
  grid::grid.newpage()
  grid::grid.draw(x)
  invisible(x)
}

set.seed(42)

#===========================================================
# helper functions
#===========================================================
#-----------------------------------------------------------
# function to generate weighted adjacency matrix -- allows more interesting graphs
#-----------------------------------------------------------
generate_weighted_adjacency_matrix <- function(n_nodes, 
                                               density = 0.3,
                                               structure = "random", 
                                               weight_distribution = "uniform",
                                               clustering = 0,
                                               community_sizes = NULL,
                                               power_alpha = 2.1) {
  
  # initialize the adjacency matrix
  adj_matrix <- matrix(0, nrow = n_nodes, ncol = n_nodes)
  
  # generate structure based on selected network type
  if (structure == "random") {
    # random Erdos–Renyi graph
    connection_probability <- matrix(runif(n_nodes * n_nodes), nrow = n_nodes)
    connection_probability <- (connection_probability + t(connection_probability))/2
    mask <- connection_probability <= density
    adj_matrix[mask] <- 1
    
  } else if (structure == "small_world") {
    # small-world network (Watts-Strogatz inspired)
    k <- max(1, round(density * n_nodes)) # Average degree
    
    # first create a ring lattice
    for (i in 1:n_nodes) {
      for (j in 1:floor(k/2)) {
        neighbor <- ((i + j - 1) %% n_nodes) + 1
        adj_matrix[i, neighbor] <- 1
        adj_matrix[neighbor, i] <- 1
        
        neighbor <- ((i - j - 1 + n_nodes) %% n_nodes) + 1
        adj_matrix[i, neighbor] <- 1
        adj_matrix[neighbor, i] <- 1
      }
    }
    
    # rewire edges with probability beta
    beta <- clustering 
    for (i in 1:n_nodes) {
      for (j in which(adj_matrix[i,] == 1)) {
        if (i < j && runif(1) < beta) {
          # rewire this edge
          adj_matrix[i, j] <- 0
          adj_matrix[j, i] <- 0
          
          # find a new target that's not already connected
          possible_targets <- which(adj_matrix[i,] == 0 & (1:n_nodes != i))
          if (length(possible_targets) > 0) {
            new_target <- sample(possible_targets, 1)
            adj_matrix[i, new_target] <- 1
            adj_matrix[new_target, i] <- 1
          }
        }
      }
    }
    
  } else if (structure == "scale_free") {
    # scale-free network (Barabási–Albert inspired)
    # start with a small connected network
    m0 <- min(5, n_nodes)
    for (i in 1:m0) {
      for (j in 1:m0) {
        if (i != j) {
          adj_matrix[i, j] <- 1
        }
      }
    }
    
    # add remaining nodes with preferential attachment
    m <- max(1, round(density * n_nodes / 2)) # Edges to add per new node
    
    for (i in (m0+1):n_nodes) {
      # calculate the degree of each existing node
      degrees <- colSums(adj_matrix[1:(i-1), 1:(i-1)])
      
      # calculate connection probabilities proportional to degree
      probs <- degrees / sum(degrees)
      
      # connect to m existing nodes without replacement
      targets <- sample(1:(i-1), min(m, i-1), prob=probs, replace=FALSE)
      adj_matrix[i, targets] <- 1
      adj_matrix[targets, i] <- 1
    }
    
  } else if (structure == "community") {
    # community structure
    if (is.null(community_sizes)) {
      # default: create roughly equal-sized communities
      n_communities <- max(2, round(sqrt(n_nodes)/2))
      community_sizes <- rep(floor(n_nodes/n_communities), n_communities)
      community_sizes[1] <- n_nodes - sum(community_sizes[-1])
    }
    
    if (sum(community_sizes) != n_nodes) {
      stop("Sum of community sizes must equal n_nodes")
    }
    
    # assign nodes to communities
    communities <- rep(1:length(community_sizes), times = community_sizes)
    
    # create dense connections within communities, sparse between communities
    within_density <- min(0.9, max(0.5, density * 3))
    between_density <- max(0.01, density / 5)
    
    for (i in 1:n_nodes) {
      for (j in i:n_nodes) {
        if (i != j) {
          # nodes in same community
          if (communities[i] == communities[j]) {
            if (runif(1) < within_density) {
              adj_matrix[i, j] <- 1
              adj_matrix[j, i] <- 1
            }
          } else {
            # nodes in different communities
            if (runif(1) < between_density) {
              adj_matrix[i, j] <- 1
              adj_matrix[j, i] <- 1
            }
          }
        }
      }
    }
  }
  
  # zero out diagonal
  diag(adj_matrix) <- 0
  
  # assign weights based on the selected distribution
  weighted_matrix <- adj_matrix
  edges <- which(adj_matrix == 1)
  
  if (weight_distribution == "uniform") {
    weighted_matrix[edges] <- runif(length(edges))
    
  } else if (weight_distribution == "normal") {
    weighted_matrix[edges] <- rnorm(length(edges), mean=0.5, sd=0.15)
    weighted_matrix[edges] <- pmax(0.01, pmin(1, weighted_matrix[edges]))
    
  } else if (weight_distribution == "power_law") {
    # generate power-law distributed weights
    weights <- runif(length(edges))^(1/(power_alpha-1))
    weighted_matrix[edges] <- weights
    
  } else if (weight_distribution == "bimodal") {
    # bimodal weight distribution: strong and weak connections
    is_strong <- runif(length(edges)) < 0.3
    weak_weights <- runif(sum(!is_strong), min=0.01, max=0.3)
    strong_weights <- runif(sum(is_strong), min=0.7, max=1)
    
    weighted_matrix[edges][!is_strong] <- weak_weights
    weighted_matrix[edges][is_strong] <- strong_weights
  }
  
  # ensure symmetry
  weighted_matrix[lower.tri(weighted_matrix)] <- t(weighted_matrix)[lower.tri(weighted_matrix)]
  
  return(weighted_matrix)
}

#-----------------------------------------------------------
# create a mice pair dataframe from adjacency matrix
#-----------------------------------------------------------
create_mice_pair_dataframe <- function(adj_matrix, id_prefix = "Mouse") {
  n_mice <- nrow(adj_matrix)
  mouse_ids <- paste0(id_prefix, "_", 1:n_mice)
  
  pairs_df <- data.frame(id1 = character(),
                         id2 = character(),
                         edge_weight = numeric(),
                         stringsAsFactors = FALSE)
  
  for (i in 1:n_mice) {
    for (j in i:n_mice) {
      if (i != j && adj_matrix[i, j] > 0) {
        pairs_df <- rbind(pairs_df, data.frame(
          id1 = mouse_ids[i],
          id2 = mouse_ids[j],
          edge_weight = adj_matrix[i, j],
          stringsAsFactors = FALSE
        ))
      }
    }
  }
  
  pairs_df <- pairs_df %>% arrange(desc(edge_weight))
  return(pairs_df)
}

#-----------------------------------------------------------
# initialize microbiome for each mouse 
#-----------------------------------------------------------
initialize_microbiomes <- function(n_mice, total_taxa = 1000, active_taxa = 300) {
  microbiomes <- matrix(0, nrow = n_mice, ncol = total_taxa)
  
  for (mouse in 1:n_mice) {
    active_indices <- sample(1:total_taxa, active_taxa)
    microbiomes[mouse, active_indices] <- runif(active_taxa, min = 1, max = 100)
    
    # normalize to 100%
    total <- sum(microbiomes[mouse, ])
    if (total > 0) {
      microbiomes[mouse, ] <- microbiomes[mouse, ] / total * 100
    }
  }
  
  return(microbiomes)
}

#-----------------------------------------------------------
# the ABM: function to run microbiome exchange simulation
# - includes competition
# - includes threshold-based extinction
# - kills taxa not exchanged by 4th exchange event for each individual
#-----------------------------------------------------------
simulate_competitive_microbiome_exchange <- function(adj_matrix,
                                                     microbiomes,
                                                     days = 30,
                                                     exchange_factor = 0.1,
                                                     extinction_threshold = 0.001) {
  n_mice <- nrow(adj_matrix)
  total_taxa <- ncol(microbiomes)
  
  microbiome_history <- list()
  microbiome_history[[1]] <- microbiomes
  
  # track exchange history for each mouse and taxon
  # exchange_count[mouse, taxon] = number of times this taxon has been exchanged for this mouse
  exchange_count <- matrix(0, nrow = n_mice, ncol = total_taxa)
  
  # track which taxa were exchanged in each day for each mouse
  # exchanged_today[mouse, taxon] = TRUE if this taxon was exchanged today for this mouse
  exchanged_today <- matrix(FALSE, nrow = n_mice, ncol = total_taxa)
  
  for (day in 1:days) {
    # copy current state
    new_microbiomes <- microbiomes
    
    # Reset exchanged_today matrix for this day
    exchanged_today <- matrix(FALSE, nrow = n_mice, ncol = total_taxa)
    
    #-----------------------------------------------------------------------
    # step A: exchange among mice (symmetric) based on adjacency & exchange_factor
    #-----------------------------------------------------------------------
    for (mouse1 in 1:(n_mice - 1)) {
      for (mouse2 in (mouse1 + 1):n_mice) {
        weight <- adj_matrix[mouse1, mouse2]
        
        if (weight > 0) {
          exchange_amount <- weight * exchange_factor
          transfer_to_mouse1 <- microbiomes[mouse2, ] * exchange_amount
          transfer_to_mouse2 <- microbiomes[mouse1, ] * exchange_amount
          
          # Update microbiomes
          new_microbiomes[mouse1, ] <- new_microbiomes[mouse1, ] + transfer_to_mouse1
          new_microbiomes[mouse2, ] <- new_microbiomes[mouse2, ] + transfer_to_mouse2
          
          # Track which taxa were exchanged (if transfer amount > 0)
          exchanged_today[mouse1, transfer_to_mouse1 > 0] <- TRUE
          exchanged_today[mouse2, transfer_to_mouse2 > 0] <- TRUE
        }
      }
    }
    
    #-----------------------------------------------------------------------
    # step B: update exchange count and kill taxa not exchanged by 4th exchange
    #-----------------------------------------------------------------------
    for (mouse in 1:n_mice) {
      # update exchange count for taxa that were exchanged today
      exchange_count[mouse, exchanged_today[mouse, ]] <- exchange_count[mouse, exchanged_today[mouse, ]] + 1
      
      # kill taxa that haven't been exchanged by the 4th exchange event
      # (i.e., taxa with exchange_count < 4) - but only after day 3
      if (day > 3) {
        taxa_to_kill <- exchange_count[mouse, ] < 4
        new_microbiomes[mouse, taxa_to_kill] <- 0
      }
      
      # threshold-based extinction
      new_microbiomes[mouse, new_microbiomes[mouse, ] < extinction_threshold] <- 0
      
      # normalize to 100%
      total <- sum(new_microbiomes[mouse, ])
      if (total > 0) {
        new_microbiomes[mouse, ] <- new_microbiomes[mouse, ] / total * 100
      }
    }
    
    # update state
    microbiomes <- new_microbiomes
    microbiome_history[[day + 1]] <- microbiomes
  }
  
  return(microbiome_history)
}

#-----------------------------------------------------------
# function to calculate Jaccard similarity between mice microbiomes
# - includes presence threshold
#-----------------------------------------------------------
calculate_jaccard_similarity_with_threshold <- function(microbiome_history,
                                                        presence_threshold = 0) {
  n_days <- length(microbiome_history)
  n_mice <- nrow(microbiome_history[[1]])
  
  jaccard_similarity <- list()
  richness <- matrix(0, nrow = n_days, ncol = n_mice)
  
  for (day in 1:n_days) {
    sim_matrix <- matrix(0, nrow = n_mice, ncol = n_mice)
    
    # compute richness
    for (mouse in 1:n_mice) {
      # presence if abundance > presence_threshold
      richness[day, mouse] <- sum(microbiome_history[[day]][mouse, ] > presence_threshold)
    }
    
    # compute pairwise Jaccard
    for (mouse1 in 1:n_mice) {
      for (mouse2 in 1:mouse1) {
        if (mouse1 == mouse2) {
          sim_matrix[mouse1, mouse2] <- 1
        } else {
          presence1 <- microbiome_history[[day]][mouse1, ] > presence_threshold
          presence2 <- microbiome_history[[day]][mouse2, ] > presence_threshold
          
          intersection <- sum(presence1 & presence2)
          union <- sum(presence1 | presence2)
          
          jaccard_val <- ifelse(union > 0, intersection / union, 0)
          
          sim_matrix[mouse1, mouse2] <- jaccard_val
          sim_matrix[mouse2, mouse1] <- jaccard_val
        }
      }
    }
    
    jaccard_similarity[[day]] <- sim_matrix
  }
  
  mean_jaccard <- numeric(n_days)
  for (day in 1:n_days) {
    lower_tri <- jaccard_similarity[[day]][lower.tri(jaccard_similarity[[day]])]
    mean_jaccard[day] <- mean(lower_tri)
  }
  
  return(list(
    jaccard_similarity = jaccard_similarity,
    mean_jaccard = mean_jaccard,
    richness = richness
  ))
}

#-----------------------------------------------------------
# visualization functions
#-----------------------------------------------------------
visualize_weighted_graph <- function(adj_matrix, node_sizes = NULL, node_colors = NULL, labels = NULL) {
  g <- graph_from_adjacency_matrix(adj_matrix, mode = "undirected", weighted = TRUE)
  
  if (is.null(node_sizes))  node_sizes <- rep(10, vcount(g))
  if (is.null(node_colors)) node_colors <- rep("skyblue", vcount(g))
  if (is.null(labels))      labels <- paste0("M", 1:vcount(g))
  
  V(g)$size <- node_sizes
  V(g)$color <- node_colors
  V(g)$label <- labels
  E(g)$width <- E(g)$weight * 5
  
  layout <- layout_with_fr(g, niter = 500)
  
  par(mar = c(0, 0, 2, 0))
  plot(g, 
       layout = layout,
       vertex.label.dist = 0.5,
       vertex.label.color = "black",
       vertex.frame.color = "gray",
       edge.color = "gray",
       edge.curved = 0.2,
       main = "mice interaction network")
  
  legend_colors <- colorRampPalette(c("lightgray", "darkgray"))(5)
  legend_labels <- seq(0.2, 1.0, length.out = 5)
  legend("bottomright", 
         legend = paste0("Weight: ", legend_labels), 
         col = legend_colors, 
         lwd = seq(1, 5, length.out = 5), 
         cex = 0.8, 
         title = "Edge Weight")
  
  return(g)
}

visualize_adjacency_matrix <- function(adj_matrix, mouse_ids) {
  melted_matrix <- melt(adj_matrix)
  colnames(melted_matrix) <- c("Row", "Column", "Value")
  melted_matrix$Row_Label <- mouse_ids[melted_matrix$Row]
  melted_matrix$Column_Label <- mouse_ids[melted_matrix$Column]
  
  ggplot(melted_matrix, aes(x = Column_Label, y = Row_Label, fill = Value)) +
    geom_tile(color = "white") +
    scale_fill_gradient(low = "white", high = "steelblue", limits = c(0, 1)) +
    theme_minimal() +
    labs(title = "mice interaction network matrix",
         x = "Mouse ID", y = "Mouse ID", fill = "Interaction\nStrength") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          plot.title = element_text(hjust = 0.5, size = 14),
          legend.title = element_text(size = 10),
          legend.text = element_text(size = 8))
}


# helper to extract a single legend from a ggplot
g_legend <- function(a.gplot) {
  tmp <- ggplot_gtable(ggplot_build(a.gplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  if (length(leg) > 0) tmp$grobs[[leg]] else NULL
}

plot_jaccard_trends <- function(jaccard_results, mouse_ids) {
  n_days <- length(jaccard_results$jaccard_similarity)
  
  # 1) mean ± 95% CI for each day
  ci_data <- data.frame(Day = 0:(n_days - 1), Mean = NA, Lower = NA, Upper = NA)
  for (d in seq_len(n_days)) {
    mat  <- jaccard_results$jaccard_similarity[[d]]
    vals <- mat[lower.tri(mat)]  # exclude diagonal
    mn   <- mean(vals)
    sem  <- sd(vals) / sqrt(length(vals))
    ci_data$Mean[d]  <- mn
    ci_data$Lower[d] <- mn - 1.96 * sem
    ci_data$Upper[d] <- mn + 1.96 * sem
  }
  
  # plot the mean Jaccard with ribbon
  p1 <- ggplot(ci_data, aes(x = Day, y = Mean)) +
    geom_ribbon(aes(ymin = Lower, ymax = Upper),
                fill = "steelblue", alpha = 0.2) +
    geom_line(color = "steelblue", size = 1.2) +
    geom_point(color = "steelblue", size = 2) +
    labs(
      title = "mean Jaccard similarity (±95% CI)",
      x = "day",
      y = "mean Jaccard similarity"
    ) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 14))
  
  # 2) we only want six snapshots (or fewer if n_days < 6)
  max_plots <- min(6, n_days)
  selected_days <- unique(round(seq(1, n_days, length.out = max_plots)))
  
  # 3) build each heatmap with identical color scale
  heatmap_list <- list()
  fill_scale <- scale_fill_gradient(low = "white", high = "steelblue", limits = c(0,1)) 
  for (day in selected_days) {
    sim_mat <- jaccard_results$jaccard_similarity[[day]]
    sim_df  <- melt(sim_mat)
    colnames(sim_df) <- c("Mouse1", "Mouse2", "Similarity")
    
    p <- ggplot(sim_df, aes(x = factor(Mouse1), y = factor(Mouse2), fill = Similarity)) +
      geom_tile() +
      fill_scale +
      labs(title = paste("Day", day - 1), x=NULL, y=NULL) +
      theme_minimal() +
      # remove axis text/ticks so no mouse IDs appear
      theme(
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        legend.title = element_text(size=9),  # optional
        legend.text  = element_text(size=8)   # optional
      )
    
    heatmap_list[[length(heatmap_list) + 1]] <- p
  }
  
  # 4) extract legend from the first plot, then remove legends from them all
  my_legend <- g_legend(heatmap_list[[1]])
  heatmap_list_no_legend <- lapply(heatmap_list, function(x) x + theme(legend.position="none"))
  
  # 5) combine all heatmaps in one panel + put the legend at the bottom
  combined_heatmaps <- arrangeGrob(
    do.call(arrangeGrob, c(heatmap_list_no_legend, ncol=3)), 
    my_legend,
    ncol = 2,               # place subplots and legend side-by-side
    widths = c(8, 1)        # make the right column narrower for the legend
  )
  
  return(list(
    mean_plot     = p1,                # line plot with CI
    heatmap_plots = combined_heatmaps  # single multi‐panel with shared legend
  ))
}


#-----------------------------------------------------------
# simulation pipeline
#-----------------------------------------------------------
run_microbiome_simulation <- function(n_mice = 20, 
                                      total_taxa = 1000, 
                                      active_taxa = 300,
                                      days_to_simulate = 30,
                                      exchange_factor = 0.001,
                                      network_density = 0.2,
                                      extinction_threshold = 0.001,
                                      presence_threshold = 0.001,
                                      structure = "random") {
  start_time <- Sys.time()
  cat("Starting microbiome ABM simulation at", format(start_time), "\n\n")
  
  # 1. generate network
  cat("Step 1: generating mice interaction network...\n")
  adj_matrix <- generate_weighted_adjacency_matrix(n_mice, structure, density = network_density)
  mouse_ids <- paste0("Mouse_", 1:n_mice)
  mice_pairs_df <- create_mice_pair_dataframe(adj_matrix, id_prefix = "Mouse")
  cat("- Created network with", n_mice, "mice and", nrow(mice_pairs_df), "connections\n")
  
  # 2. initialize microbiomes
  cat("\nStep 2: initializing microbiomes for each mouse...\n")
  initial_microbiomes <- initialize_microbiomes(n_mice, total_taxa, active_taxa)
  cat("- Each mouse initialized with", active_taxa, "taxa from a pool of", total_taxa, "possible taxa\n")
  
  # 3. run simulation with competition & threshold
  cat("\nStep 3: running microbiome exchange simulation...\n")
  cat("- Simulating", days_to_simulate, "days with exchange factor =", exchange_factor, "\n")
  microbiome_history <- simulate_competitive_microbiome_exchange(
    adj_matrix,
    initial_microbiomes,
    days = days_to_simulate,
    exchange_factor = exchange_factor,
    extinction_threshold = extinction_threshold
  )
  cat("- Simulation complete\n")
  
  # 4. calculate Jaccard similarity using presence threshold
  cat("\nStep 4: calculating Jaccard similarity metrics...\n")
  jaccard_results <- calculate_jaccard_similarity_with_threshold(microbiome_history,
                                                                 presence_threshold)
  
  # initial and final average similarity
  initial_avg_sim <- jaccard_results$mean_jaccard[1]
  final_avg_sim   <- jaccard_results$mean_jaccard[days_to_simulate + 1]
  cat("- Initial average Jaccard similarity:", round(initial_avg_sim, 4), "\n")
  cat("- Final average Jaccard similarity:", round(final_avg_sim, 4), "\n")
  cat("- Change in similarity:", round(final_avg_sim - initial_avg_sim, 4), "\n")
  
  # 5. visualizations
  cat("\nStep 5: generating visualizations...\n")
  cat("- Visualizing mice interaction network\n")
  network_heatmap <- visualize_adjacency_matrix(adj_matrix, mouse_ids)
  (g <- visualize_weighted_graph(adj_matrix, labels = mouse_ids))
  
  cat("- Visualizing Jaccard similarity trends\n")
  jaccard_plots <- plot_jaccard_trends(jaccard_results, mouse_ids)
  
  # 6. create final pairwise dataframe with edge_weight + final-day Jaccard
  cat("\nStep 6: creating final pairwise data frame...\n")
  final_day_index <- days_to_simulate + 1
  final_sim_mat <- jaccard_results$jaccard_similarity[[final_day_index]]
  
  final_pairs_df <- data.frame(
    mouseid_1       = character(),
    mouseid_2       = character(),
    edge_weight     = numeric(),
    outcome_variable= numeric(),
    stringsAsFactors = FALSE
  )
  
  for (i in 1:(n_mice - 1)) {
    for (j in (i + 1):n_mice) {
      final_pairs_df <- rbind(
        final_pairs_df,
        data.frame(
          mouseid_1        = mouse_ids[i],
          mouseid_2        = mouse_ids[j],
          edge_weight      = adj_matrix[i, j],
          outcome_variable = final_sim_mat[i, j],
          stringsAsFactors = FALSE
        )
      )
    }
  }
  
  # 7. create density plot of outcome variable (Jaccard similarity)
  final_jaccard_density_plot <- ggplot(final_pairs_df, aes(x = outcome_variable)) +
    geom_density(fill = "steelblue", alpha = 0.4) +
    labs(title = "Density of final-day Jaccard similarity",
         x = "Jaccard similarity", y = "Density") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 14))
  
  # total time
  end_time <- Sys.time()
  execution_time <- difftime(end_time, start_time, units = "secs")
  cat("\nSimulation completed in", round(execution_time, 2), "seconds\n")
  
  # return results
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
    mice_pairs          = mice_pairs_df,
    microbiome_history  = microbiome_history,
    jaccard_results     = jaccard_results,
    network_heatmap     = network_heatmap,
    network_graph       = g,
    jaccard_plots       = jaccard_plots,
    final_pairs_df              = final_pairs_df,
    final_jaccard_density_plot  = final_jaccard_density_plot
  ))
}




