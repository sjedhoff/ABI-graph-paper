################################################################################
##                                                                            ##
##                     Case Study Mice: Real world data                       ##
##                              Data Preparation                              ##
##                                                                            ##
################################################################################


################################################################################
##                              Microbiome data                               ##
################################################################################
### data from https://catalogue.ceh.ac.uk/documents/043513e5-406c-4477-89aa-c96059acb232
### info table from https://www.nature.com/articles/s41559-024-02381-0

microbiome_data <- readRDS("case_study_mice/real_data/Wytham_HH_processed_microbiome_data.rds")

tax_tab <- tax_table(microbiome_data)
genus <- as.vector(tax_tab[,"Genus"])

otu_tab <- otu_table(microbiome_data)

sample_tab <- sample_data(microbiome_data)

key <- data.frame(ID = sample_tab$Individual_ID, 
                  sample_name = sample_tab$Sample_name)

### Infos
infos_taxa <- readr::read_delim("case_study_mice/real_data/supp_table_4.csv", 
                         delim = ";", escape_double = FALSE, trim_ws = TRUE)


### Social graph: anaerobic non-sporeformers 
social_taxa <- infos_taxa[infos_taxa$Aerotolerance_binary == "anaerobic" & 
                            infos_taxa$Spore_formation == "NSF","Genus"]
idy <- unlist(sapply(social_taxa$Genus, function(x) which(grepl(x, genus, ignore.case = TRUE))))

microbiome_social <- otu_tab[,idy]
microbiome_social <- microbiome_social[,which(unname(apply(microbiome_social, 2, sum)) != 0)]
microbiome_social <- data.frame("sample_name" = rownames(microbiome_social), unname(microbiome_social))
microbiome_social <- merge(microbiome_social, key, by = "sample_name", all = TRUE)
microbiome_social <- microbiome_social[c(1,ncol(microbiome_social),2:(ncol(microbiome_social)-1))]

microbiome_social_agg <- microbiome_social %>%
  group_by(ID) %>%
  summarise(
    across(X1:X820, ~ mean(.x, na.rm = TRUE)),
    .groups = "drop"
  )

microbiome_social_mice <- microbiome_social_agg %>%
  rowwise() %>%
  mutate(
    s = sum(c_across(X1:X820)),
    across(X1:X820, ~ ifelse(s == 0, 0, .x / s * 100))
  ) %>%
  ungroup() %>%
  select(-s)

saveRDS(microbiome_social_mice, file = "case_study_mice/real_data/data_ready/social_taxa_df.rds")


################################################################################
##                  Social Distances from Logger tag data                     ##
################################################################################
### Code from https://github.com/nuorenarra/Analysing-dyadic-data-with-brms

library(asnipe)
Loggerdata <- readr::read_csv("case_study_mice/real_data/Wytham_HH_Logger_tag_data.csv")
Loggerdata_reduced <- Loggerdata[,c("ID","X_coord","Y_coord","LOGGER_ID")]
head(Loggerdata_reduced)

# Introduce night variable
dt <- as.POSIXct(Loggerdata$datetime, format = "%Y-%m-%d %H:%M", tz = "Europe/Berlin")
night <- as.Date(dt - 12*60*60)
Loggerdata$night <- as.integer(factor(night))

# Make a spatio-temporal grouping variable by combining spatial and temporal variables "logger" (=unique location) and "night" (=unique time).
Loggerdata$night_logger <- paste(Loggerdata$night, Loggerdata$LOGGER_ID, sep="-")
# Aggregate Loggerdata per individual
t <- table(Loggerdata$ID, Loggerdata$night_logger)
log <- t > 0
# Make "group-by-individual matrix" (gbi), where lognight-logger is considered the grouping variable. All individuals present in each spatio-temporal combination are considered part of the same "group". The more two individuals are observed in the same "group" the more socially associated they are.
gbi <- replace(t, log, 1)
gbi <- t(gbi) # Here individuals as columns and groups (nights_logger combinations) as rows
#derive social association matrix using the default Simple Ratio Index method of asnipe package
AM <- get_network(gbi, association_index="SRI")

saveRDS(AM, file = "case_study_mice/real_data/data_ready/social_distances_matrix.rds")



################################################################################
##                              Social graph                                  ##
################################################################################

AM <- readRDS(file = "case_study_mice/real_data/data_ready/social_distances_matrix.rds")
microbiome_social_mice <- readRDS(file = "case_study_mice/real_data/data_ready/social_taxa_df.rds")

ids <- microbiome_social_mice$ID
microbiome_social_sub <- microbiome_social_mice[microbiome_social_mice$ID %in% colnames(AM),]
AM_sub <- AM[colnames(AM) %in% microbiome_social_mice$ID, colnames(AM) %in% microbiome_social_mice$ID]
social_graph <- cbind(microbiome_social_sub[,-1], AM_sub)

saveRDS(social_graph, file = "case_study_mice/real_data/data_ready/social_graph.rds")
saveRDS(microbiome_social_sub, file = "case_study_mice/real_data/data_ready/social_taxa_df_sub.rds")
saveRDS(AM_sub, file = "case_study_mice/real_data/data_ready/social_distances_matrix_sub.rds")


# select taxa with largest variances and re-normalize
taxa_selected <- names(which(apply(microbiome_social_sub[,-1], 2, var) > 2))
microbiome_social_selected <- microbiome_social_sub[,c("ID", taxa_selected)]
microbiome_social_selected <- microbiome_social_selected %>%
  rowwise() %>%
  mutate(
    s = sum(c_across(taxa_selected)),
    across(taxa_selected, ~ ifelse(s == 0, 0, .x / s * 100))
  ) %>%
  ungroup() %>%
  select(-s)

social_graph_selected <- cbind(microbiome_social_selected[,-1], AM_sub)

saveRDS(social_graph_selected, file = "case_study_mice/real_data/data_ready/social_graph_selected.rds")
