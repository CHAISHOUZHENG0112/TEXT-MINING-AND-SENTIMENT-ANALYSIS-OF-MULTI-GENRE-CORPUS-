# --------------------------------------------------
# R script: TEXT MINING AND SENTIMENT ANALYSIS OF MULTI-GENRE CORPUS 
# Project: TEXT MINING AND SENTIMENT ANALYSIS OF MULTI-GENRE CORPUS
#
# Date: 14/5/2025
# Time: 2:14AM
# Author: CHAI SHOU ZHENG
# File: "Assignment3_Corpus2"
# --------------------------------------------------


# clean up the environment before starting
rm(list = ls())

# Load libraries
library(slam)
library(tm)
library(SnowballC)
library(proxy)
library(dplyr)
library(SentimentAnalysis)
library(igraph)


setwd("FIT3152/FIT3152 Ass3")
getwd()


# --------------------------------------------------
# Q3. Document-Term Matrix (DTM)
# --------------------------------------------------
cname = file.path(".", "Assignment3_Corpus2") # Set path to the text corpus folder 
print(dir(cname))  # Print file names in the folder

text_corpus = Corpus(DirSource(cname, recursive = TRUE)) # Load all text files into a text corpus
summary(text_corpus) # Show summary of the corpus (number of docs, etc.)


# ----------- Text Cleaning and Preprocessing -----------

# Tokenisation
text_corpus <- tm_map(text_corpus, removeNumbers)                      # Remove numbers
text_corpus <- tm_map(text_corpus, removePunctuation)                  # Remove punctuation
text_corpus <- tm_map(text_corpus, content_transformer(tolower))       # Convert all text to lowercase

# Filter words
# Define extra stopwords to remove less informative words
extra_stopwords <- c("use", "get", "make", "need", "just", "can",
                     "also", "one", "will", "time", "like", "may", 
                     "much", "more", "that", "with",
                     "arent", "doesnt", "dont", "even", "lot", 
                     "might", "often", "wont", "cant", "still")
# Combine built-in stopwords with custom ones
all_stopwords <- c(stopwords("english"), extra_stopwords)  

text_corpus <- tm_map(text_corpus, removeWords, all_stopwords)         # Remove common stopwords like "the", "and", "is"
text_corpus <- tm_map(text_corpus, stripWhitespace)                    # Remove extra whitespace
text_corpus <- tm_map(text_corpus, stemDocument, language = "english") # Apply stemming (reduce words to root form)



# ----------- Create and Inspect Document-Term Matrix -----------
dtm <- DocumentTermMatrix(text_corpus) # Create DTM from cleaned corpus
dtm                                    # View DTM
dim(dtm)                               # Print dimensions: number of documents x number of terms
inspect(dtm)                           # Show part of the DTM

# ----------- Trim Sparse Terms -----------
dtm_trimmed <- removeSparseTerms(dtm, 0.65) # Remove sparse terms 
dtm_trimmed           # View trimmed DTM

dim(dtm_trimmed)      # Show dimensions after trimming
inspect(dtm_trimmed)  # View sample of the trimmed DTM

colnames(as.matrix(dtm_trimmed)) # Show the remaining terms after trimming

# Find terms that appear in at least 10 documents
findFreqTerms(dtm_trimmed, lowfreq = 10)

# Convert DTM to a data frame for inspection/analysis
dtm_df <- as.data.frame(as.matrix(dtm_trimmed))   # Convert DTM to data frame
dtm_df <- dtm_df[, order(colnames(dtm_df))]       # Sort columns
print(dtm_df)  # Print final DTM data frame


# --------------------------------------------------
# Q4. Hierarchical clustering and Dendrogram
# --------------------------------------------------
set.seed(34035958)

# Convert the trimmed DTM to a matrix
dtm_matrix <- as.matrix(dtm_trimmed)

# Create cosine distance matrix using the trimmed dtm with 23 tokens
distmatrix <- proxy::dist(dtm_matrix, method = "cosine")

# Perform hierarchical clustering using Ward's method
fit <- hclust(distmatrix, method = "ward.D")

# Plot the dendrogram
plot(fit, 
     hang = -1, 
     main = "Hierarchical Clustering of Assignment3_Corpus2 (Using Trimmed DTM with 23 Tokens)",
     xlab = "Documents", 
     sub = "", 
     cex = 0.8)

# Draw rectangles around k clusters (e.g., k = 3 for 3 genres)
rect.hclust(fit, k = 3, border = "blue")


# --------------------------------------------------
# Evaluate Clustering Accuracy
# --------------------------------------------------
# Create actual genre labels: 7 Car, 7 Smartphone, 7 Sports
topics <- c(rep("Car", 7), rep("Smartphone", 7), rep("Sports", 7))

# Cut dendrogram into 3 groups (clusters)
groups <- cutree(fit, k = 3)

# Compare predicted clusters with actual genre labels
conf_matrix <- table(GroupNames = topics, Clusters = groups)
print(conf_matrix)

# Reorganize confusion matrix and calculate accuracy
TA <- as.data.frame.matrix(conf_matrix)
TA <- TA[, c(2, 1, 3)]             # Rearrange columns for better alignment
correct <- sum(apply(TA, 1, max))  # Largest value per row
total <- sum(TA)                   # Total documents
accuracy <- correct / total        # Clustering accuracy
print(paste("Clustering Accuracy:", round(accuracy * 100, 2), "%"))


# --------------------------------------------------
# Analyze Most Frequent Terms per Genre
# --------------------------------------------------
# Add genre labels to the DTM
genres <- c(rep("Car_Reviews", 7), rep("Smartphone_Reviews", 7), rep("Sports_Articles", 7))
dtm_df$Genre <- genres

# Sum term frequencies by genre
genre_term_freq <- dtm_df %>%
  group_by(Genre) %>%
  summarise(across(everything(), sum))

# Function to get top N terms for a genre
top_n_terms <- function(data, genre_label, n = 10) {
  # Get the row that matches the genre
  row_vector <- as.numeric(data[data$Genre == genre_label, -1])  # remove Genre column
  names(row_vector) <- colnames(data)[-1]  # assign term names
  sorted <- sort(row_vector, decreasing = TRUE)
  return(head(sorted, n))
}

# Print top 10 terms for each genre
top_n_terms(genre_term_freq, "Car_Reviews")
top_n_terms(genre_term_freq, "Smartphone_Reviews")
top_n_terms(genre_term_freq, "Sports_Articles")

# --------------------------------------------------
# Export Augmented DTM
# --------------------------------------------------
# Add Cluster labels to the DTM data
aug_dtms <- cbind(dtm_df, Genre = genres, Cluster = groups)

# Sort by Cluster 
aug_dtms <- aug_dtms[order(aug_dtms$Cluster), ]

# Save the result to CSV
write.csv(aug_dtms, "Augmented_DTM.csv", row.names = TRUE)


# --------------------------------------------------
# Q5. Sentiment Analysis
# --------------------------------------------------
cname = file.path(".", "Assignment3_Corpus2")  

text_corpus = Corpus(DirSource(cname, recursive = TRUE))

# Perform sentiment analysis directly on corpus
SentimentA <- analyzeSentiment(text_corpus)
SentimentA$Genre <- c(rep("Car_Reviews", 7), rep("Smartphone_Reviews", 7), rep("Sports_Articles", 7))

# Set layout to 1 row, 3 columns
par(mfrow = c(1, 3))

# Plot the three GI-based sentiment measures by Genre
boxplot(SentimentGI ~ Genre, data = SentimentA, main = "SentimentGI by Genre")
boxplot(PositivityGI ~ Genre, data = SentimentA, main = "PositivityGI by Genre")
boxplot(NegativityGI ~ Genre, data = SentimentA, main = "NegativityGI by Genre")


# Set up 2x2 plot layout
par(mfrow = c(2, 2))

# Plot sentiment metrics grouped by Genre
boxplot(WordCount ~ Genre, data = SentimentA, frame = TRUE, main = "Word Count by Genre")
boxplot(SentimentQDAP ~ Genre, data = SentimentA, frame = TRUE, main = "Sentiment (QDAP) by Genre")
boxplot(PositivityQDAP ~ Genre, data = SentimentA, frame = TRUE, main = "Positivity (QDAP) by Genre")
boxplot(RatioUncertaintyLM ~ Genre, data = SentimentA, frame = TRUE, main = "Uncertainty Ratio (LM) by Genre")


###########################
# t-test SentimentQDAP
###########################

# Car_Reviews vs Smartphone_Reviews
lhs = SentimentA[SentimentA$Genre == "Car_Reviews", "SentimentQDAP"]
rhs = SentimentA[SentimentA$Genre == "Smartphone_Reviews", "SentimentQDAP"]
t.test(lhs, rhs, alternative = "greater")

# Car_Reviews vs Sports_Articles
lhs = SentimentA[SentimentA$Genre == "Car_Reviews", "SentimentQDAP"]
rhs = SentimentA[SentimentA$Genre == "Sports_Articles", "SentimentQDAP"]
t.test(lhs, rhs, alternative = "greater")

# Smartphone_Reviews vs Sports_Articles
lhs = SentimentA[SentimentA$Genre == "Smartphone_Reviews", "SentimentQDAP"]
rhs = SentimentA[SentimentA$Genre == "Sports_Articles", "SentimentQDAP"]
t.test(lhs, rhs, alternative = "greater")


###########################
# t-test NegativityGI
###########################

# Car_Reviews vs Smartphone_Reviews
lhs = SentimentA[SentimentA$Genre == "Car_Reviews", "NegativityGI"]
rhs = SentimentA[SentimentA$Genre == "Smartphone_Reviews", "NegativityGI"]
t.test(lhs, rhs, alternative = "greater")

# Car_Reviews vs Sports_Articles
lhs = SentimentA[SentimentA$Genre == "Car_Reviews", "NegativityGI"]
rhs = SentimentA[SentimentA$Genre == "Sports_Articles", "NegativityGI"]
t.test(lhs, rhs, alternative = "greater")

# Smartphone_Reviews vs Sports_Articles
lhs = SentimentA[SentimentA$Genre == "Smartphone_Reviews", "NegativityGI"]
rhs = SentimentA[SentimentA$Genre == "Sports_Articles", "NegativityGI"]
t.test(lhs, rhs, alternative = "greater")



# --------------------------------------------------
# Q6. Create a single-mode network (documents)
# --------------------------------------------------
###########################
# plot basic network graph
###########################

set.seed(34035958)

# Start with trimmed DTM matrix
dtmsx <- as.matrix(dtm_trimmed)

# Convert to binary matrix (1 if word appears in document, 0 otherwise)
dtmsx_binary <- (dtmsx > 0) + 0

# Create adjacency matrix by multiplying with transpose
ByAbsMatrix <- dtmsx_binary %*% t(dtmsx_binary)

# Remove self-links (no edge from document to itself)
diag(ByAbsMatrix) <- 0

# Create an undirected, weighted graph
ByAbsGraph <- graph_from_adjacency_matrix(ByAbsMatrix, mode = "undirected", weighted = TRUE)

# Plot the graph
plot(ByAbsGraph, vertex.label.cex = 0.8, edge.width = E(ByAbsGraph)$weight)


####################################################
# Identify Most Connected Documents (Degree Centrality)
####################################################

degree_centrality <- degree(ByAbsGraph)
sorted_degree <- sort(degree_centrality, decreasing = TRUE)
print("Top documents by Degree Centrality:")
print(sorted_degree)


###############################
# plot improved network graph
###############################

set.seed(34035958)

# Create matrix from trimmed DTM
dtmsx <- as.matrix(dtm_trimmed)

# Convert to binary matrix (1 if word appears, else 0)
dtmsx_binary <- (dtmsx > 0) + 0

# Create adjacency matrix by multiplying with transpose
ByAbsMatrix <- dtmsx_binary %*% t(dtmsx_binary)

# Remove self-connections
diag(ByAbsMatrix) <- 0

# Create undirected, weighted graph
ByAbsGraph <- graph_from_adjacency_matrix(ByAbsMatrix, mode = "undirected", weighted = TRUE)

# Assign genre based on document names
genre <- ifelse(grepl("Car_review", rownames(ByAbsMatrix)), "Car",
                ifelse(grepl("Smartphone_review", rownames(ByAbsMatrix)), "Smartphone", "Sports"))

# Assign colors to genres
genre_colors <- c("Car" = "skyblue", "Smartphone" = "orange", "Sports" = "green")
vertex_colors <- genre_colors[genre]

# Scale node size based on degree centrality
degree_vals <- degree(ByAbsGraph)
vertex_sizes <- 5 + (degree_vals / max(degree_vals)) * 10  # Scale from 5 to 15

# Fix layout with set.seed for reproducibility
layout_orig <- layout_with_fr(ByAbsGraph)

# Plot improved graph
plot(ByAbsGraph,
     layout = layout_orig,
     vertex.color = vertex_colors,
     vertex.size = vertex_sizes,
     vertex.label.cex = 0.7,
     vertex.label.color = "black",
     edge.width = E(ByAbsGraph)$weight,
     main = "Improved Document Network by Genre")

# Add legend
legend("topright",
       legend = c("Car Reviews", "Smartphone Reviews", "Sports Articles"),
       col = genre_colors,
       pch = 19,
       pt.cex = 2,
       bty = "n")


# --------------------------------------------------
# Q7. Create a single-mode network (tokens)
# --------------------------------------------------
###########################
# plot basic network graph
###########################

set.seed(123456)

# Convert DTM to matrix 
dtmsx <- as.matrix(dtm_trimmed)

# Convert to binary matrix (1 = token appears in doc, 0 = does not)
dtmsx_binary <- (dtmsx > 0) + 0

# Create Token-Token adjacency matrix
ByTokenMatrix <- t(dtmsx_binary) %*% dtmsx_binary

# Remove self-links (no term connected to itself)
diag(ByTokenMatrix) <- 0

# Create undirected, weighted token graph
TokenGraph <- graph_from_adjacency_matrix(ByTokenMatrix, mode = "undirected", weighted = TRUE)

# Basic plot of the token network
plot(TokenGraph,
     vertex.label.cex = 0.7,
     edge.width = E(TokenGraph)$weight,
     main = "Basic Token Network Graph")



####################################################
# Identify Most Connected Token (Degree Centrality)
####################################################

# Calculate Degree Centrality for tokens
degree_centrality_tokens <- degree(TokenGraph)

# Step 4: Sort and print top tokens
sorted_degree_tokens <- sort(degree_centrality_tokens, decreasing = TRUE)
print("Top tokens by Degree Centrality:")
print(sorted_degree_tokens)


########################################################
# Identify Most Connected Token (eigenvector centrality)
#########################################################

# Calculate eigenvector centrality
central_tokens <- eigen_centrality(TokenGraph)$vector

# View top 5 most central tokens
head(sort(central_tokens, decreasing = TRUE), 5)



###############################
# plot improved network graph
###############################
set.seed(1234)

# Convert DTM to matrix
dtmsx <- as.matrix(dtm_trimmed)

# Convert to binary matrix 
dtmsx_binary <- (dtmsx > 0) + 0

# Create Token-Token adjacency matrix
ByTokenMatrix <- t(dtmsx_binary) %*% dtmsx_binary

# Remove self-links
diag(ByTokenMatrix) <- 0

# Create undirected, weighted token graph
TokenGraph <- graph_from_adjacency_matrix(ByTokenMatrix, mode = "undirected", weighted = TRUE)

# Compute degree centrality
degree_vals <- degree(TokenGraph)

# Highlight top 10 tokens by degree in red
top_tokens <- names(sort(degree_vals, decreasing = TRUE)[1:10])
vertex_colors <- ifelse(V(TokenGraph)$name %in% top_tokens, "tomato", "skyblue")

# Scale node sizes by degree
vertex_sizes <- 5 + (degree_vals / max(degree_vals)) * 10  # Scale 5–15

# Use a consistent layout
layout_token <- layout_with_fr(TokenGraph)

# Plot the improved graph
plot(TokenGraph,
     layout = layout_token,
     vertex.color = vertex_colors,
     vertex.size = vertex_sizes,
     vertex.label.cex = 0.7,
     vertex.label.color = "black",
     edge.width = E(TokenGraph)$weight,
     main = "Improved Token Network with Top Tokens Highlighted")



# --------------------------------------------------
# Q8. Create a bipartite (two-mode) network 
# --------------------------------------------------

##########################
# Transform data format
##########################

# Convert DTM to data frame and add document IDs
dtmsa <- as.data.frame(as.matrix(dtm_trimmed))
dtmsa$doc_id <- rownames(dtmsa)

# Reshape to long format
dtmsb <- data.frame()
for (i in 1:nrow(dtmsa)) {
  for (j in 1:(ncol(dtmsa) - 1)) {
    touse <- cbind(dtmsa[i, j], dtmsa[i, ncol(dtmsa)], colnames(dtmsa)[j])
    dtmsb <- rbind(dtmsb, touse)
  }
}
colnames(dtmsb) <- c("weight", "doc_id", "token")

# Remove zero-weight entries
dtmsc <- dtmsb[dtmsb$weight != 0, ]
dtmsc <- dtmsc[, c("doc_id", "token", "weight")]


#################################
# Plot the basic bipartite graph
#################################

# Create igraph bipartite network
g <- graph.data.frame(dtmsc, directed = FALSE)
V(g)$type <- bipartite_mapping(g)$type
V(g)$color <- ifelse(V(g)$type, "lightgreen", "pink")  # Docs vs Tokens
V(g)$shape <- ifelse(V(g)$type, "circle", "square")
E(g)$color <- "grey"

# Plot the basic bipartite graph
set.seed(123)

plot(g,
     vertex.label.cex = 0.7,
     vertex.size = 5,
     edge.width = E(g)$weight,
     main = "Bipartite Network:Documents and Tokens")


####################################################
# Identify Most Connected Token (Degree Centrality)
####################################################

# Compute degree centrality
degree_vals <- degree(g)

# Get token nodes (the ones NOT ending in ".txt")
token_nodes <- V(g)[!grepl("\\.txt$", names(V(g)))]

# Top 10 most connected tokens
top_tokens <- sort(degree_vals[token_nodes], decreasing = TRUE)[1:10]
print(top_tokens)


#################################
# Identify Least Connected Token
#################################
# Filter only document nodes (type == TRUE)
doc_nodes <- V(g)[V(g)$type == TRUE]

# Get documents with the lowest degree (least connected)
low_doc_degrees <- sort(degree_vals[doc_nodes], decreasing = FALSE)[1:10]
print(low_doc_degrees)


#################################
# Plot the improved bipartite graph
#################################

# Create igraph bipartite network
g <- graph.data.frame(dtmsc, directed = FALSE)

# Define node types: TRUE = document, FALSE = token
V(g)$type <- bipartite_mapping(g)$type

# Assign genres for document nodes
genres <- rep("Token", vcount(g))  # default to "Token"
names(genres) <- V(g)$name         # match by name!

# Assign genre based on name pattern
genres[grepl("^Car_review.*\\.txt$", V(g)$name)] <- "Car"
genres[grepl("^Smartphone_review.*\\.txt$", V(g)$name)] <- "Smartphone"
genres[grepl("^Sports_article.*\\.txt$", V(g)$name)] <- "Sports"

# Assign colors to genres
genre_colors <- c("Car" = "skyblue", "Smartphone" = "orange", "Sports" = "lightgreen", "Token" = "tomato")
V(g)$color <- genre_colors[genres[V(g)$name]]

# Assign shape by type
V(g)$shape <- ifelse(V(g)$type, "circle", "square")  # docs = circle, tokens = square

# Size by degree
degree_vals <- degree(g)
V(g)$size <- ifelse(V(g)$type, 8 + degree_vals * 0.3, 4 + degree_vals * 0.2)

# Label setup
V(g)$label <- V(g)$name
V(g)$label.cex <- 0.6

# Edge styling
E(g)$color <- "grey"
E(g)$width <- as.numeric(E(g)$weight) * 1.2

# Plot the graph
set.seed(123)
plot(g,
     layout = layout_with_fr(g),
     vertex.label = V(g)$label,
     vertex.label.cex = V(g)$label.cex,
     vertex.label.color = "black",
     vertex.size = V(g)$size,
     vertex.color = V(g)$color,
     vertex.shape = V(g)$shape,
     edge.color = E(g)$color,
     edge.width = E(g)$width,
     main = "Bipartite Network: Documents and Tokens")

# Add legend
legend("topleft",
       legend = c("Car Reviews (doc)", "Smartphone Reviews (doc)", "Sports Articles (doc)", "Tokens"),
       col = c("skyblue", "orange", "lightgreen", "tomato"),
       pch = c(19, 19, 19, 15),  # 19 = circle (doc), 15 = square (token)
       pt.cex = 1.5,
       bty = "n")

