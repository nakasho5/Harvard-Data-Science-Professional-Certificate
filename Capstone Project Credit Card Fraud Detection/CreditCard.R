# Author: Sho Nakamura
# Title: Detect Credit Card Fraud Project


##########################################################
# Import libraries
##########################################################
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(caTools)) install.packages("caTools", repos = "http://cran.us.r-project.org")
if(!require(pROC)) install.packages("pROC", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(rpart.plot)) install.packages("rpart.plot", repos = "http://cran.us.r-project.org")
if(!require(gbm)) install.packages("gbm", repos = "http://cran.us.r-project.org")
if(!require(ranger)) install.packages("ranger", repos = "http://cran.us.r-project.org")
if(!require(Epi)) install.packages("Epi", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(ggplot2)
library(caret)
library(data.table)
library(dplyr)
library(corrplot)
library(caTools)
library(pROC)
library(rpart)
library(rpart.plot)
library(gbm, quietly=TRUE)
library(ranger)
library(Epi)

# Set seed
set.seed(1996)


##########################################################
# Data exploration and visualization
##########################################################
# Load dataset
# Due to the large size of the dataset, it could not be uploaded to Github.
# Since the link is posted on Github, please download from there.
credit_card <- read.csv("creditcard.csv")

# Check dataset
head(credit_card, 5)
tail(credit_card, 5)
names(credit_card)

summary(credit_card)

# Check fraud counts
table(credit_card$Class)

# Check proportion of classes
prop.table(table(credit_card$Class))

# Check the missing values
colSums(is.na(credit_card))

var(credit_card$Amount)
sd(credit_card$Amount)

# Plot distribution of class labels
credit_card %>%
  ggplot(aes(x = factor(Class), fill = factor(Class))) +
  geom_bar() +
  ggtitle("Distribution of Class Labels")
  
# Plot distribution of time of transaction by class
credit_card %>%
  ggplot(aes(x = Time, fill = factor(Class))) +
  geom_histogram(bins = 100) +
  ggtitle("Distribution of time of transaction by class") +
  labs(x = "Time (seconds)", y = "Number of Transactions") +
  facet_grid(Class ~ ., scales = 'free_y')

# Plot correlation
corr <- cor(credit_card, use = "pairwise.complete.obs")
corrplot(corr, tl.col = "black")


##########################################################
# Build model
##########################################################
# Create dataset
credit_card$Amount <- scale(credit_card$Amount)
data <- credit_card[, -1]
head(data)

# Split train and test data
split <- sample.split(data$Class, SplitRatio = 0.80)
train <- subset(data, split == TRUE)
test <- subset(data, split == FALSE)
dim(train)
dim(test)


## Logistic_Regression_Model ##
log_model <- glm(Class ~ ., train, family = "binomial")
log_pred <- predict(log_model, test, probability = TRUE)
roc(test$Class, log_pred, plot = TRUE, col = "red")



## Decision Tree ##
dt_model <- rpart(Class ~ ., train, method = "class")
dt_pred <- predict(dt_model, test, type = "class")
rpart.plot(dt_model)
ROC(test$Class, dt_pred, plot = "ROC")



## Gradient Boosting (GBM) ##
system.time(
  gbm_model <- gbm(Class ~ .,
                   distribution = "bernoulli",
                   data = rbind(train, test),
                   n.trees = 500,
                   interaction.depth = 3,
                   n.minobsinnode = 100,
                   shrinkage = 0.01,
                   bag.fraction = 0.5,
                   train.fraction = nrow(train) / (nrow(train) + nrow(test))
 )
)
gbm.iter <- gbm.perf(gbm_model, method = "test")
model.influence = relative.influence(gbm_model, n.trees = gbm.iter, sort. = TRUE)
gbm_test <- predict(gbm_model, newdata = test, n.trees = gbm.iter)
gbm_auc <- roc(test$Class, gbm_test, plot = TRUE, col = "red")
print(gbm_auc)