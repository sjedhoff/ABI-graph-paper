################################################################################
##                                                                            ##
##                  Case Study Trains: Simulator function                     ##
##                                                                            ##
################################################################################


### Train Simulator:
#  Input:
#   - adj: adjacency matrix of connected sections of the train network
#   - schedule: matrix, each column is a train and each row contains the number 
#       of section the train should go on
#   - section_stats: data.frame: each row is information about one section
#             - default_time: default time of this section
#             - prob_zero_delay: probability, that on this section is zero delay
#             - delay_lambda: parameter of the exponential distribution for the possible delay
train_simulator <- function(adj, schedule, section_stats){
  #------ Delay Calculator Function --------------------------------------------#
  
  ### Input:
  #   - pi: probability, that the section has 0 delay
  #   - lambda: parameters of the normal distribution, if the section has delay
  random_delay_time <- function(pi, lambda){
    is_zero <- sample(x = c(TRUE, FALSE), size = 1, prob = c(pi, 1-pi))
    if(is_zero){
      return(0)
    }
    else{
      return(round(rgamma(1, shape = 5, rate = lambda)))
    }
  }
  
  
  #------ Simulation ----------------------------------------------------------#
  
  n_trains <- ncol(schedule)
  n_stops <- nrow(schedule)
  time_without_delay <- apply(schedule, 2, function(x) sum(section_stats[x, "default_time"]))
  n_rows <- length(time_without_delay)
  
  delay_names <- paste0("time_with_delay_s", seq_len(n_stops))
  delay_df <- as.data.frame(matrix(0, nrow = n_rows, ncol = n_stops,
                                   dimnames = list(NULL, delay_names)))
  
  time_log <- cbind(
    data.frame(time_without_delay = time_without_delay),
    delay_df
  )
  
  
  all_trains_done <- FALSE
  
  # State at t=0
  current_state <- schedule[1,]
  state_count <- rep(1, n_trains)
  time_left <- as.numeric(section_stats[current_state, "default_time"] + 
                            apply(section_stats[current_state, c("prob_zero_delay", "delay_lambda")], 1, function(x) random_delay_time(x[1], x[2])))
  time_log$time_with_delay_s1 <- time_left
  
  t <- 0
  
  while(!all_trains_done){
    t <- t+1
    if(t > 500){
      return(list(rep(NA, n_trains),
                  graph))
    }
    # in the next minute
    #print(current_state)
    prev_state <- current_state
    
    
    # check if a train is in state 4 (the goal)
    if(any(state_count >= n_stops, na.rm = TRUE)){
      train_done <- which(state_count == n_stops)
      # remove train from all active status vectors
      time_left[train_done] <- NA
      current_state[train_done] <- NA
      state_count[train_done] <- NA
      
      # check if any trains are left
      if(all(is.na(state_count))){
        all_trains_done <- TRUE
        next
      }
    }
    
    # if all trains have still time left on their section, move trains along their current section
    if(all(time_left > 0, na.rm=TRUE)){
      time_left <- time_left - 1
    }
    # if at least one train has to switch sections
    else{
      # for train which moves sections: check if section is free
      train_id <- which(time_left <= 0)
      next_section <- unlist(lapply(train_id, function(train) schedule[state_count[train] + 1, train]))
      
      # check which trains can move and which not
      train_stuck <- train_id[next_section %in% current_state]
      train_moving <- train_id[!(next_section %in% current_state)]
      
      # if two trains want to move to the same section at the same time
      next_track <- next_section[train_moving]
      if(any(duplicated(next_track))){
        duplicated_track <- next_section[duplicated(next_track)]
        duplicated_trains <- train_moving[which(next_track %in% duplicated_track)]
        train_stuck_2 <- duplicated_trains[which.min(time_left[duplicated_trains])]
        train_stuck <- sort(c(train_stuck, train_stuck_2))
        train_moving <- train_moving[! train_moving %in% train_stuck]
      }
      
      
      move_trains <- TRUE
      while(move_trains){
        # for the trains who can move
        if(length(train_moving) > 0){
          state_count[train_moving] <- state_count[train_moving] + 1
          next_section_train_moving <- next_section[which(train_id %in% train_moving)]
          # get new time_left
          time_for_section <- section_stats[next_section_train_moving, "default_time"] +
            apply(section_stats[next_section_train_moving, c("prob_zero_delay", "delay_lambda")], 1, function(x) random_delay_time(x[1], x[2]))
          time_left[train_moving] <- time_for_section
          time_log[cbind(train_moving, state_count[train_moving]+1)] <- time_log[cbind(train_moving, state_count[train_moving]+1)] + time_for_section
          # update current state
          current_state[train_moving] <- next_section_train_moving
          train_moving <- numeric(0)
        }
        
        # check if stuck_trains can move now:
        if(length(train_stuck) > 0){#
          #print(str(train_stuck))
          next_track <- schedule[cbind(state_count[train_stuck]+1, train_stuck)]
          
          if( all(sort(current_state[train_stuck]) == sort(next_track))){
            train_moving <- train_stuck
            train_stuck <- numeric(0)
          }
          else{
            # multiple trains want to move to same track
            if(any(duplicated(next_track))){
              duplicated_track <- next_track[duplicated(next_track)] 
              duplicated_trains <- which(next_track %in% duplicated_track)
              train_moving <- train_stuck[which.min(time_left[train_stuck][duplicated_trains])]
              train_stuck <- train_stuck[!train_stuck %in% train_moving]
            }
            else{
              train_moving <- train_stuck[!next_section[which(train_id %in% train_stuck)] %in% current_state]
              train_stuck <- train_stuck[next_section[which(train_id %in% train_stuck)] %in% current_state]
            }
            
          }
          
          if(length(train_moving) == 0){
            move_trains <- FALSE
          }
          
        }
        
        else{
          move_trains <- FALSE
        }
        
      }
      
      # for the trains who can not move
      if(length(train_stuck) > 0){
        #debug()
        time_left[train_stuck] <- time_left[train_stuck] - 1 
        time_log[cbind(train_stuck, state_count[train_stuck] + 1)] <- time_log[cbind(train_stuck, state_count[train_stuck] + 1)] + 1
      }
      # move other trains along their section
      time_left[-train_id] <- time_left[-train_id] - 1
    }
    
  }
  
  time_log$total_time_with_delay <- rowSums(time_log[,-1])
  
  
  #------ Graph Structure for Bayesflow ---------------------------------------#
  
  
  n_tracks <- nrow(section_stats)                          
  schedule_dummy <- matrix(0L, nrow = n_tracks, ncol = length(schedule))
  schedule_dummy[cbind(as.vector(schedule), seq_len(length(schedule)))] <- 1L
  
  graph <- cbind(section_stats$default_time, schedule_dummy)
  
  inference_parameters <- time_log$total_time_with_delay + rnorm(4, 0, 1)
  
  return(list(inference_parameters,
              graph))
  
}

