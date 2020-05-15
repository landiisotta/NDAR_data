require(ggplot2)
require(reshape2)
library(RColorBrewer)
require(ComplexHeatmap)
require(ggalluvial)

#########################FUNCTIONS#############################
# Compare missing information
compare_missinfo <- function(p, lev){
  print(sprintf('Check missing data confounders period %s, subdomains', p))
  percDf <- read.table(file.path(folder_name, 
                                 paste('VINELANDmiss_perc_', lev, p, '.csv', sep='')),
                       header=TRUE,
                       as.is=TRUE,
                       sep=',')
  percDf$cluster <- as.factor(percDf$cluster)
  
  for (col in names(percDf)[2:(length(names(percDf))-1)]){
    print(col)
    if (length(unique(percDf$cluster)) == 2){
      print(t.test(percDf[, col]~percDf$cluster))
    } else{
      print(summary(aov(percDf[, col]~percDf$cluster)))
      print(pairwise.t.test(percDf[, col], percDf$cluster))}
    }
}

# Check for confounders: SEX, SITE, PHENO
confounders <- function(df, p, lev){
  if (lev == 'L1'){
    # SEX
    print('Compare sex for subdomain clusters')
    tabsex_sub <- table(df$sex, df$cluster_subdomain)
    print(chisq.test(tabsex_sub))
    
    # SITE
    print('Compare sites for subdomain clusters')
    tabsite_sub <- table(df$site, df$cluster_subdomain)
    print(chisq.test(tabsite_sub))
    
    # PHENOTYPE
    print('Compare phenotypes for subdomain clusters')
    tabpheno_sub <- table(df$phenotype, df$cluster_subdomain)
    print(chisq.test(tabpheno_sub))
    
    # INTERVIEW AGE
    if (length(unique(df$cluster_subdomain))==2){
      print(t.test(df$interview_age~df$cluster_subdomain))
    } else{
      print(summary(aov(interview_age~cluster_subdomain, df)))
      print(pairwise.t.test(df$interview_age, df$cluster_subdomain))}
  } else {
    #SEX
    print('Compare sex for domain clusters')
    tabsex_dom <- table(df$sex, df$cluster_domain)
    print(chisq.test(tabsex_dom))
    
    #SITE
    print('Compare sites for domain clusters')
    tabsite_dom <- table(df$site, df$cluster_domain)
    print(chisq.test(tabsite_dom))
    
    #PHENO
    print('Compare phenotype for domain clusters')
    tabpheno_dom <- table(df$phenotype, df$cluster_domain)
    print(chisq.test(tabpheno_dom))
    
    #AGE
    if (length(unique(df$cluster_domain))==2){
      print(t.test(df$interview_age~df$cluster_domain))
    } else{
      print(summary(aov(interview_age~cluster_domain, df)))
      print(pairwise.t.test(df$interview_age, df$cluster_domain))
      }
  }
  }
  
# Compare clusters
clust_comparison <- function(df, p, lev){
  if (lev == 'L1'){
  sprintf('Comparing subdomain scores at period $s', p)
  df_long <- melt(subset(df, select=c('subjectkey', 'cluster_subdomain', subdomain_features)), 
                  id.vars=c('subjectkey', 'cluster_subdomain'))
  df_long$cluster_subdomain <- as.character(df_long$cluster_subdomain)
  print(ggplot(df_long, aes(x=variable, y=value, fill=cluster_subdomain)) +
          geom_boxplot() +
          facet_wrap(~variable, scale="free") +
          coord_cartesian(ylim = c(min(df_long$value), max(df_long$value))) +
          ggtitle(sprintf('Subdomain features (VINELAND) -- period %s', p)))
  
  for (col in subdomain_features){
    print(col)
    if (length(unique(df$cluster_subdomain))==2){    
      print(t.test(df[, col]~df$cluster_subdomain, alternative='less'))
    } else{
      print(summary(aov(df[, col]~df$cluster_subdomain)))
      print(pairwise.t.test(df[, col], df$cluster_subdomain))}
  }
  } else {
  sprintf('Comparing domain scores at period %s', p)
  df_long <- melt(subset(df, select=c('subjectkey', 'cluster_domain', domain_features)), 
                  id.vars=c('subjectkey', 'cluster_domain'))
  df_long$cluster_domain <- as.character(df_long$cluster_domain)
  print(ggplot(df_long, aes(x=variable, y=value, fill=cluster_domain)) +
          geom_boxplot() +
          facet_wrap(~variable, scale="free") +
          coord_cartesian(ylim = c(min(df_long$value), max(df_long$value))) +
          ggtitle(sprintf('Domain features (VINELAND) -- period %s', p)))
  
  for (col in domain_features){
    print(col)
    if (length(unique(df$cluster_domain))==2){    
      print(t.test(df[, col]~df$cluster_domain, alternative='less'))
    } else{
      print(summary(aov(df[, col]~df$cluster_domain)))
      print(pairwise.t.test(df[, col], df$cluster_domain))}
  }
  }}

# Compare features within the same cluster
# Compare scores within clusters
feat_comparison <- function(df, p, lev){
  if (lev=='L1'){
  df_long <- melt(subset(df, select=c('subjectkey', 'cluster_subdomain', subdomain_features)), 
                  id.vars=c('subjectkey', 'cluster_subdomain'))
  df_long$cluster_subdomain <- as.character(df_long$cluster_subdomain)
  for (cl in sort(unique(df_long$cluster_subdomain))){
    print(sprintf('Analyzing cluster %s', cl))
    print(pairwise.t.test(df_long$value[which(df_long$cluster_subdomain==cl)], 
                          df_long$variable[which(df_long$cluster_subdomain==cl)]))
  }
  print(ggplot(df_long, aes(x=cluster_subdomain, y=value, fill=variable)) +
          geom_boxplot() +
          facet_wrap(~cluster_subdomain, scale="free") +
          coord_cartesian(ylim = c(min(df_long$value), max(df_long$value))) +
          ggtitle(sprintf('Subdomain features for each clusters (VINELAND) -- period %s', p)))} else{
  df_long <- melt(subset(df, select=c('subjectkey', 'cluster_domain', domain_features)), 
                  id.vars=c('subjectkey', 'cluster_domain'))
  df_long$cluster_domain <- as.character(df_long$cluster_domain)
  for (cl in sort(unique(df_long$cluster_domain))){
    print(sprintf('Analyzing cluster %s', cl))
    print(pairwise.t.test(df_long$value[which(df_long$cluster_domain==cl)], 
                          df_long$variable[which(df_long$cluster_domain==cl)]))
  }
  print(ggplot(df_long, aes(x=cluster_domain, y=value, fill=variable)) +
          geom_boxplot() +
          facet_wrap(~cluster_domain, scale="free") +
          coord_cartesian(ylim = c(min(df_long$value), max(df_long$value))) +
          ggtitle(sprintf('Domain features for each clusters (VINELAND) -- period %s', p))) 
          }
}

# Heatmaps for replicability
# Visualize subject distance between train/test and within clusters
replheat <- function(p, lev){
  # TRAIN
  distdf_tr <- read.table(file.path(folder_name, 
                                 paste('VINELAND_dist', toupper(lev), 'TR', p, '.csv', sep='')),
                       header = TRUE,
                       as.is = TRUE,
                       sep = ',',
                       row.names=1)
  if (lev=='subdomain'){
  clust_tr <- distdf_tr$cluster_subdomain
  distmat_tr <- as.matrix(subset(distdf_tr, select=-c(cluster_subdomain)))
  } else{
  clust_tr <- distdf_tr$cluster_domain
  distmat_tr <- as.matrix(subset(distdf_tr, select=-c(cluster_domain)))
  }
  
  row.names(distmat_tr) <- row.names(distdf_tr)
  colnames(distmat_tr) <- names(distdf_tr)[1:(ncol(distdf_tr)-1)]
  
  colSide <- brewer.pal(9, "Set1")[3:9]
  col_v <- list(clusters = c())
  for (idx in sort(unique(clust_tr))){
  col_v$clusters <- c(col_v$clusters, colSide[idx])}
  names(col_v$clusters) <- as.character(sort(unique(clust_tr)))
  
  hTR <- Heatmap(distmat_tr,
               heatmap_legend_param = list(
                 title = paste('VINELAND', '\ndist mat TR', sep=''), at = seq(min(distmat_tr),
                                                                            max(distmat_tr), 0.5)),
               # name = paste(name_ins, '\ndist mat TR', sep=''),
               show_row_names = FALSE,
               show_column_names = FALSE,
               show_row_dend = FALSE,
               show_column_dend = FALSE,
               cluster_rows = FALSE,
               cluster_columns = FALSE,
               # col = colorRampPalette(brewer.pal(8, "Blues"))(25),
               left_annotation = HeatmapAnnotation(clusters=clust_tr,
                                                   col=col_v, which='row'),
               top_annotation = HeatmapAnnotation(clusters=clust_tr,
                                                  col=col_v, which='column', show_legend = FALSE))
  # TEST
  distdf_ts <- read.table(file.path(folder_name, 
                                  paste('VINELAND_dist', toupper(lev), 'TS', p, '.csv', sep='')),
                        header = TRUE,
                        as.is = TRUE,
                        sep = ',',
                        row.names=1)
  if (lev=='subdomain'){
  clust_ts <- distdf_ts$cluster_subdomain
  distmat_ts <- as.matrix(subset(distdf_ts, select=-c(cluster_subdomain)))
  } else{
  clust_ts <- distdf_ts$cluster_domain
  distmat_ts <- as.matrix(subset(distdf_ts, select=-c(cluster_domain)))
  }
  
  row.names(distmat_ts) <- row.names(distdf_ts)
  colnames(distmat_ts) <- names(distdf_ts)[1:(ncol(distdf_ts)-1)]
  
  col_vts <- list(clusters = c())
  for (idx in sort(unique(clust_ts))){
  col_vts$clusters <- c(col_vts$clusters, colSide[idx])}
  names(col_vts$clusters) <- as.character(sort(unique(clust_ts)))
  
  hTS <- Heatmap(distmat_ts,
               heatmap_legend_param = list(
                 title = paste('VINELAND', '\ndist mat TS', sep=''), at = seq(min(distmat_ts),
                                                                              max(distmat_ts), 0.5)),
               # name = paste(name_ins, '\ndist mat TR', sep=''),
               show_row_names = FALSE,
               show_column_names = FALSE,
               show_row_dend = FALSE,
               show_column_dend = FALSE,
               cluster_rows = FALSE,
               cluster_columns = FALSE,
               # col = colorRampPalette(brewer.pal(8, "Blues"))(25),
               left_annotation = HeatmapAnnotation(clusters=clust_ts,
                                                   col=col_vts, which='row'),
               top_annotation = HeatmapAnnotation(clusters=clust_ts,
                                                  col=col_vts, which='column', show_legend = FALSE))
  grid.newpage()
  title = sprintf('%s feature Level %s distance matrices train/test comparisons', 'VINELAND', lev)
  grid.text(title, x=unit(0.5, 'npc'), y=unit(0.8, 'npc'), just='centre')
  pushViewport(viewport(x = 0, y = 0.75, width = 0.5, height = 0.5, just = c("left", "top")))
  grid.rect(gp = gpar(fill = "#00FF0020"))
  draw(hTR, newpage = FALSE)
  popViewport()
  
  pushViewport(viewport(x = 0.5, y = 0.75, width = 0.5, height = 0.5, just = c("left", "top")))
  grid.rect(gp = gpar(fill = "#0000FF20"))
  draw(hTS, newpage = FALSE)
  popViewport()
  }

## Pos-hoc comparisons, replication of clusters, new clustering
## Vineland L1, L2 levels at P1, P2, P3
folder_name <- './out'

domain_features <- c('communicationdomain_totalb',
                   'livingskillsdomain_totalb', 
                   'socializationdomain_totalb')
subdomain_features <- c('receptive_vscore', 'expressive_vscore',
                      'personal_vscore', 'domestic_vscore',
                      'community_vscore', 'interprltn_vscore', 'playleis_vscore',
                      'copingskill_vscore')

# Read missing data patterns
for (p in c('P1', 'P2', 'P3')){
  
  # Results replicability 
  replheat(p, 'subdomain')
  replheat(p, 'domain')
  
  # Compare missing info
  # compare_missinfo(p, 'L1')
  # compare_missinfo(p, 'L2')
  
  # df <- read.table(file.path(folder_name, 
  #                            paste('VINELANDdata', p, '.csv', sep='')),
  #                  header=TRUE,
  #                  as.is=TRUE,
  #                  sep=',')
  # # Confounders
  # confounders(df, p, 'L1')
  # confounders(df, p, 'L2')
  # 
  # # Compare instrument mean scores
  # clust_comparison(df, p, 'L1')
  # clust_comparison(df, p, 'L2')
  # 
  # # Compare feature scores within the same cluster
  # feat_comparison(df, p, 'L1')
  # feat_comparison(df, p, 'L2')
  # 
  # # Plot alluvial plot
  # sprintf('Alluvial plot for period %s between Vineland domains and subdomains', p)
  # alldf <- subset(df, select=c(subjectkey, cluster_subdomain, cluster_domain, sex))
  # alldf <- alldf[order(alldf$sex),]
  # print(is_alluvia_form(alldf))
  # plot(ggplot(alldf,
  #             aes(axis1 = cluster_subdomain, axis2 = cluster_domain)) +
  #        geom_alluvium(aes(fill=sex), width = 1/12) +
  #        geom_stratum(width = 1/12, fill = "black", color = "grey") +
  #        geom_label(stat = "stratum", infer.label = TRUE) +
  #        scale_x_discrete(limits = c("Level 1", "Level 2"), expand = c(.05, .05)) +
  #        scale_fill_brewer(type = "qual", palette = "Set1") +
  #        ggtitle(sprintf('Subject movements between Vineland subdomains and domains at period %s', p)))
  # 
  # df$cluster <- paste(df$cluster_subdomain, df$cluster_domain, sep='-')
}

  
  

