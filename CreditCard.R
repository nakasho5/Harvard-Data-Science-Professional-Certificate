# Author: Sho Nakamura
# Title: Detect Credit Card Fraud Project


##########################################################
# Import libraries
##########################################################
library(tidyverse)
library(ggplot2)
library(dplyr)
library(corrplot)
library(caret)
library(data.table)
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