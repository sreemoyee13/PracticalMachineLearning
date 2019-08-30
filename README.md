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
The plot is available in [Corr _Plot.pdf] file in the GitHub repository  (https://github.com/sreemoyee13/PracticalMachineLearning/blob/gh-pages/Corr_Plot.pdf)

```r
highlyCorrelated = findCorrelation(cor_mat, cutoff=0.75)
names(trainData)[highlyCorrelated]
[1] "accel_belt_z"      "roll_belt"         "accel_belt_y"     
 [4] "total_accel_belt"  "accel_dumbbell_z"  "accel_belt_x"     
 [7] "pitch_belt"        "magnet_dumbbell_x" "accel_dumbbell_y" 
[10] "magnet_dumbbell_y" "accel_dumbbell_x"  "accel_arm_x"      
[13] "accel_arm_z"       "magnet_arm_y"      "magnet_belt_z"    
[16] "accel_forearm_y"   "gyros_forearm_y"   "gyros_dumbbell_x" 
[19] "gyros_dumbbell_z"  "gyros_arm_x"
```
```r
decisionTreeMod1 <- rpart(classe ~ ., data=trainData, method="class")
fancyRpartPlot(decisionTreeMod1)
View(decisionTreeMod1)

```
The plot is available in the [Decision _Tree.png] file in the GitHub repository  (https://github.com/sreemoyee13/PracticalMachineLearning/blob/gh-pages/Decision%20_Tree.png)

```r
predictTreeMod1 <- predict(decisionTreeMod1, testData, type = "class")
> cmtree <- confusionMatrix(predictTreeMod1, testData$classe)
> cmtree
Confusion Matrix and Statistics
 
      	Reference
Prediction	A	B    C	D	E
     	A 1067  105    9   24	9
     	B   40  502   59   63   77
     	C   28   90  611  116   86
     	D   11   49   41  423   41
     	E   19   41   18   46  548
 
Overall Statistics
                                     	
               Accuracy : 0.7642     	
             	95% CI : (0.751, 0.7771)
    No Information Rate : 0.2826     	
	P-Value [Acc > NIR] : < 2.2e-16  	
                                     	
                  Kappa : 0.7015     	
                                     	
 Mcnemar's Test P-Value : < 2.2e-16  	
 
Statistics by Class:
 
                     Class: A Class: B Class: C Class: D
Sensitivity            0.9159   0.6379   0.8279   0.6295
Specificity            0.9503   0.9284   0.9055   0.9589
Pos Pred Value         0.8789   0.6775   0.6563   0.7487
Neg Pred Value         0.9663   0.9157   0.9602   0.9300
Prevalence             0.2826   0.1909   0.1790   0.1630
Detection Rate         0.2588   0.1218   0.1482   0.1026
Detection Prevalence   0.2944   0.1797   0.2258   0.1370
Balanced Accuracy      0.9331   0.7831   0.8667   0.7942
                     Class: E
Sensitivity            0.7201
Specificity            0.9631
Pos Pred Value         0.8155
Neg Pred Value         0.9383
Prevalence             0.1846
Detection Rate 	    0.1329
Detection Prevalence   0.1630
Balanced Accuracy      0.8416

> plot(cmtree$table, col = cmtree$byClass, 
     main = paste("Decision Tree - Accuracy =", round(cmtree$overall['Accuracy'], 4)))
```     
     
```r

> controlRF <- trainControl(method="cv", number=3)
 model_RF <- train(classe ~ ., data=trainData, method="rf", trControl=controlRF)
> print(model_RF)
Random Forest 

13737 samples
   52 predictor
    5 classes: 'A', 'B', 'C', 'D', 'E' 

No pre-processing
Resampling: Cross-Validated (3 fold) 
Summary of sample sizes: 9158, 9159, 9157 
Resampling results across tuning parameters:

  mtry  Accuracy   Kappa    
   2    0.9880617  0.9848959
  27    0.9882072  0.9850824
  52    0.9806361  0.9755043

Accuracy was used to select the optimal model using the largest value.
The final value used for the model was mtry = 27.
```
```r
> plot(model_RF,main="Accuracy Rate of RFMA with predictors")
> trainpred <- predict(model_RF,newdata=testData)
> confMatRF <- confusionMatrix(testData$classe,trainpred)
> confMatRF$table
      	Reference
Prediction	A    B	C	D    E
     	A 1165	0    0	0	0
     	B    0  787	0    0	0
         C	0	0  738	0	0
     	D    0	0	0  672	0
     	E    0	0	0    0  761
> confMatRF$overall[1]
Accuracy
   	1
> names(model_RF$finalModel)
 [1] "call"        	"type"        	"predicted"  	
 [4] "err.rate"        "confusion"   	"votes"      	
 [7] "oob.times"   	"classes"     	"importance" 	
[10] "importanceSD"    "localImportance" "proximity"  	
[13] "ntree"       	"mtry"        	"forest"     	
[16] "y"           	"test"     	   "inbag"      	
[19] "xNames"      	"problemType" 	"tuneValue"  	
[22] "obsLevels"       "param"      	
> model_RF$finalModel$classes
[1] "A" "B" "C" "D" "E"
```

```r
> plot(model_RF$finalModel,main="Error of RFMA with  trees")
> TopMostVars <- varImp(model_RF)
> TopMostVars
rf variable importance

  only 20 most important variables shown (out of 52)

                     Overall
roll_belt             100.00
pitch_forearm          61.11
yaw_belt               53.33
magnet_dumbbell_z      45.28
magnet_dumbbell_y      44.34
roll_forearm           44.13
pitch_belt             43.54
accel_dumbbell_y       22.87
roll_dumbbell          18.37
accel_forearm_x        16.99
magnet_dumbbell_x      16.58
magnet_belt_z          15.40
accel_belt_z           14.84
magnet_forearm_z       14.74
accel_dumbbell_z       14.13
total_accel_dumbbell   14.01
magnet_belt_y          12.36
gyros_belt_z           12.15
yaw_arm                10.57
magnet_belt_x          10.26
> 
```

```r

model_GBM <- train(classe~., data=trainData, method="gbm", trControl=trControl, verbose=FALSE)
> print(model_GBM)

Stochastic Gradient Boosting
 
13737 samples
   52 predictor
	5 classes: 'A', 'B', 'C', 'D', 'E'
 
No pre-processing
Resampling: Cross-Validated (3 fold)
Summary of sample sizes: 9157, 9158, 9159
Resampling results across tuning parameters:
 
  interaction.depth  n.trees  Accuracy   Kappa	
  1               	50  	0.7500893  0.6832357
  1              	100  	0.8168436  0.7681969
  1              	150  	0.8501841  0.8104436
  2               	50  	0.8541882  0.8152475
  2              	100  	0.9057278  0.8806923
  2              	150  	0.9280040  0.9088911
  3               	50  	0.8936438  0.8653523
  3              	100  	0.9395785  0.9235519
  3              	150  	0.9582877  0.9472293
 
Tuning parameter 'shrinkage' was held constant at a value
of 0.1
Tuning parameter 'n.minobsinnode' was held constant
at a value of 10
Accuracy was used to select the optimal model using the
largest value.
The final values used for the model were n.trees =
150, interaction.depth = 3, shrinkage = 0.1 and
n.minobsinnode = 10.

```

```r

> plot(model_GBM)
> trainpred <- predict(model_GBM,newdata=testData)
> confMatGBM <- confusionMatrix(testData$classe,trainpred)
 > confMatGBM$table
      	Reference
Prediction	A	B	C	D	E
     	A 1158	7	0	0	0
     	B   23  752   12	0	0
     	C	0   20  709	8	1
     	D	0	0   17  653	2
     	E	1	4	3	7  746
     	
> confMatGBM$overall[1]

Accuracy
0.9745331

> FinalPred <- predict(model_RF,newdata=validData)
> FinalPred
[1] B A B A A E D B A A B C B A E E A B B B
Levels: A B C D E
```


## I hope this fucking works
