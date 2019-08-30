---
title: "Practical Machine Learning Course Project"
author: "Sreem the Metal Queen"
date: "8/29/2019"
output: html_document
---

## R Markdown

Kill me in my face

``` r
library(caret)
library(rpart)library(RColorBrewer)
library(randomForest)
library(rattle)
library(corrplot)
library(gbm)

```

``` r
train_in <- read.csv('./pml-training.csv', header=T)
valid_in <- read.csv('./pml-testing.csv', header=T)
> dim(train_in)
[1] 19622   160
> dim(valid_in)
[1]  20 160
> trainData<- train_in[, colSums(is.na(train_in)) == 0]
> validData <- valid_in[, colSums(is.na(valid_in)) == 0]
> dim(trainData)
[1] 19622    93
> dim(validData)
[1] 20 60
> trainData <- trainData[, -c(1:7)]
> validData <- validData[, -c(1:7)]
> dim(trainData)
[1] 19622    86
> dim(validData)
[1] 20 53
```
```r
> set.seed(1234)
> inTrain <- createDataPartition(trainData$classe, p = 0.7, list = FALSE)
> trainData <- trainData[inTrain, ]
> testData <- trainData[-inTrain, ]
> dim(trainData)
[1] 13737    86
> dim(testData)
[1] 4123   86
```

```r
> NZV <- nearZeroVar(trainData)
> trainData <- trainData[, -NZV]
> testData  <- testData[, -NZV]
> dim(trainData)
[1] 13737    53
> dim(testData)
[1] 4123   53
```
```r
cor_mat <- cor(trainData[, -53])
corrplot(cor_mat, order = "FPC", method = "color", type = "upper",
tl.cex = 0.8, tl.col = rgb(0, 0, 0))
```
## I hope this fucking works
