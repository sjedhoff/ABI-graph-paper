################################################################################
##                                                                            ##
##             Case Study Trains: Schedule and Graph Simulator                ##
##                                                                            ##
################################################################################

# need to add section_stat details if duration is not the same for all sections
schedule_simulator <- function(adj, n_trains = 4, n_stops = 4, overlaps = FALSE){
  schedule_done <- FALSE
  while(!schedule_done){
    schedule <- matrix(NA, nrow = n_stops, ncol = n_trains)
    n_section <- nrow(adj)
    # randomize starting points
    start_section <- sample(1:n_section, size = n_trains, replace = FALSE)
    schedule[1,] <- start_section
    
    for(train in 1:n_trains){
      for(t in 2:n_stops){
        current_section <- schedule[t-1,train]
        connected_to_current_section <- which(adj[current_section,] == 1)
        if(overlaps){
          possible_next_sections <- connected_to_current_section
        }
        else{
          possible_next_sections <- connected_to_current_section[! connected_to_current_section %in% schedule[t,]]
          if(length(possible_next_sections) == 0){
            break
          }
        }
        next_section <- ifelse(length(possible_next_sections) > 1, sample(possible_next_sections, size = 1), possible_next_sections)
        schedule[t, train] <- next_section
      }
    }
    if(!any(is.na(schedule))){
      schedule_done <- TRUE
    }
    
  }
  
  return(schedule)
  
}

library(igraph)
adjacency_simulator <- function(n_sections = 10){
  is_done <- FALSE
  while(!is_done){
    graph <- sample_gnp(n = n_sections, p = 0.5, directed = FALSE)
    adj <- as_adjacency_matrix(graph, sparse = FALSE)
    # check if any section is not connected to anything
    if(all(rowSums(adj) != 0)){
      is_done <- TRUE
    }
  }
  
  return(adj)
}

