# Author: Sho Nakamura
# Title: Movie lens Project

##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(ggplot2)
library(data.table)
library(knitr)
library(recosystem)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 1000)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


##########################################################
# Data exploration and visualization
##########################################################

# Check the first 5 rows of the data set 
head(edx)

# Check the data set
summary(edx)

# Check the number of unique users and movies
edx %>% summarize(n_users = n_distinct(userId), n_movies = n_distinct(movieId))

# Check the relationship between userId and rating
edx %>%
  count(userId) %>%
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = 'blue') +
  scale_x_log10() +
  ggtitle("Relationship between userId and rating") +
  labs(x = "userId", y = "Number of ratings")

# Check the relationship between movieId and rating
edx %>%
  count(movieId) %>%
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = 'blue') +
  scale_x_log10() +
  ggtitle("Relationship between movieId and rating") +
  labs(x = "movieId", y = "Number of ratings")

# Check the rating histogram
edx %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating)) %>%
  ggplot(aes(b_u)) +
  geom_histogram(bins = 30, color = 'blue') +
  ggtitle("Average ratings") +
  labs(x = "Rating", y = "Number of ratings")

# Check the relationship between genre and rating
genres <- edx$genres %>% str_replace("\\|.*","") %>% unique()

nb_genres <- sapply(genres, function(x){
  index <- str_which(edx$genres, x)
  length(edx$rating[index])
})

genres_ratings <- sapply(genres, function(x){
  index <- str_which(edx$genres, x)
  mean(edx$rating[index], na.rm = T)
})

genres_table <- data.frame(genres = genres, n_genres = nb_genres, avg_rating = genres_ratings)

genres_table %>% ggplot(aes(x= reorder(genres,avg_rating), y = avg_rating)) + 
  geom_col()+ labs(x=" Genre", y="Average Rating") + 
  ggtitle("Relationship between rating and genre 1") + 
  coord_flip()

genres_table %>% ggplot(aes(x= reorder(genres,n_genres), y = n_genres)) + 
  geom_col()+ labs(x="Genre", y="Number of ratings") + 
  ggtitle("Relationship between ratings and genre 2") + 
  coord_flip()

# Check the distribution of ratings
hist(edx$rating, main="Distribution of ratings", xlab="Ratings")

##########################################################
# Build model
##########################################################

# Create train set and test set
test_index <- createDataPartition(y = edx$rating, times = 1,
                                  p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# RMSE Function
RMSE <- function(true_ratings = NULL, predicted_ratings = NULL) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

## 1. Naive Baseline Model ##
# Compute the mean rating of the dataset
mu <- mean(train_set$rating)
mu

# Test the results
nb_rmse <- RMSE(test_set$rating, mu)

# Save prediction to data frame
rmse_results <- tibble(Model = "Naive Baseline Model", RMSE = nb_rmse)
rmse_results


## 2. Movie Effect Model ##
movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

predicted_ratings <- mu + test_set %>%
  left_join(movie_avgs, by = 'movieId') %>%
  pull(b_i)

# Test the results
me_rmse <- RMSE(predicted_ratings, test_set$rating)

# Save prediction to data frame
rmse_results <- bind_rows(rmse_results, tibble(Model = "Movie Effect Model", RMSE = me_rmse))
rmse_results


## 3. Movie and User Effect Model ##
user_avgs <- train_set %>%
  left_join(movie_avgs, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- test_set %>%
  left_join(movie_avgs, by = 'movieId') %>%
  left_join(user_avgs, by = 'userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Test the results
m_u_rmse <- RMSE(predicted_ratings, test_set$rating)

# Save prediction to data frame
rmse_results <- bind_rows(rmse_results, tibble(Model = "Movie and User Effect Model", RMSE = m_u_rmse))
rmse_results


## 4. Regularization Model ##
lambdas <- seq(0, 10, 0.25)
best_lambda <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu) / (n() + l))
  b_u <- train_set %>%
    left_join(b_i, by = 'movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu) / (n() + l))
  predicted_ratings <- test_set %>%
    left_join(b_i, by = 'movieId') %>%
    left_join(b_u, by = 'userId') %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  return(RMSE(predicted_ratings, test_set$rating))
})

# Find the optimal lambda 
lambda <- lambdas[which.min(best_lambda)]
lambda

# Save prediction to data frame
rmse_results <- bind_rows(rmse_results, tibble(Model = "Regularization Model", RMSE = min(best_lambda)))
rmse_results


## 5. Matrix Factorization Model ##
set.seed(1)
train_data <-  with(train_set, data_memory(user_index = userId, 
                                           item_index = movieId,
                                           rating = rating,
                                           date = date))

test_data  <-  with(test_set,  data_memory(user_index = userId, 
                                           item_index = movieId, 
                                           rating = rating,
                                           date = date))

# Build recommender
r <-  Reco()

r$train(train_data)

predicted_ratings <-  r$predict(test_data, out_memory()) 
matrix_rmse <- RMSE(test_set$rating, predicted_ratings)

# Save prediction to data frame
rmse_results <- bind_rows(rmse_results, tibble(Model = "Matrix Factorization Model (Final Holdout Test)", RMSE = matrix_rmse))
rmse_results


##########################################################
# Final 
##########################################################
final_data  <-  with(final_holdout_test,  data_memory(user_index = userId, 
                                                   item_index = movieId, 
                                                   rating = rating,
                                                   date = date))

# predict final data set
predicted_ratings <- r$predict(final_data, out_memory()) 
final_rmse <- RMSE(final_holdout_test$rating, predicted_ratings)
rmse_results <- bind_rows(rmse_results, tibble(Model = "Matrix factorization (Final)", RMSE = final_rmse))
rmse_results