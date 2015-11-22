##Reading in the data.

download.file(url="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
              ,destfile = "./training.csv")

download.file(url="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
              ,destfile = "./testing.csv")

training.full<-read.csv("./training.csv")

testing<-read.csv("./testing.csv")

library(caret)

##Setting the seed for reproducibility.
set.seed(125)

##Splitting the training set into two sets.
inTrain.1<-createDataPartition(y=training.full$classe,p=0.80,list=FALSE)

training.1<-training.full[inTrain.1,]
training.2<-training.full[-inTrain.1,]

##Removing all NAs columns from the testing set because they can't be
##used effectively by most models. Dealing with NAs is beyond the scope
##of this analysis.
testing<-testing[ , colSums(is.na(testing)) == 0]
testing<-testing[,-60]

##Going back to the training sets and making sure they contain those 59 vars.
variables<-c("classe",colnames(testing))

idx<-match(variables,names(training.1))
training.1<-training.1[,idx]
training.2<-training.2[,idx]


accuracy<-function(model)
{
    predictions<-predict(model,training.2)
    
    return(confusionMatrix(predictions,training.2$classe)$overall[1])
}


##There are a few variables we can drop from all analyses due to their nature.
model.lda<-train(classe~.,method="lda",data=training.1[,-(2:8)],verbose=FALSE)
model.qda<-train(classe~.,method="qda",data=training.1[,-(2:8)],verbose=FALSE)
model.rpart<-train(classe~.,method="rpart",data=training.1[,-(2:8)])
model.treebag<-train(classe~.,method="treebag",data=training.1[,-(2:8)])
model.rf<-train(classe~.,method="rf",data=training.1[,-(2:8)],verbose=FALSE)
model.gbm<-train(classe~.,method="gbm",data=training.1[,-(2:8)],verbose=FALSE)


accuracy(model.rf)

accuracy(model.treebag)

accuracy(model.gbm)

accuracy(model.qda)

accuracy(model.lda)

accuracy(model.rpart)

all.accuracies<-c(lda.accuracy,qda.accuracy,gbm.accuracy,rpart.accuracy,
                  treebag.accuracy,rf.accuracy)

names(all.accuracies)<-c("LDA","QDA","GBM","RPART","TREEBAG","RF")

##Accuracy distributions.
accuracy.dist<-function(model)
{
    accuracy.dist<-vector()
    
    for(i in 1:1000){
    training.2.sample<-training.2[sample(nrow(training.2),20,replace=T),]
    
    predictions<-predict(model,training.2.sample)
    
    accuracy.dist<-c(accuracy.dist,
          confusionMatrix(predictions,training.2.sample$classe)$overall[1])
    }
    
    return(accuracy.dist)
}

accuracy.dist.lda<-accuracy.dist(model.lda)

accuracy.dist.qda<-accuracy.dist(model.qda)

accuracy.dist.gbm<-accuracy.dist(model.gbm)

accuracy.dist.rf<-accuracy.dist(model.rf)

accuracy.dist.rpart<-accuracy.dist(model.rpart)

accuracy.dist.treebag<-accuracy.dist(model.treebag)

accuracy.dists.all<-cbind(accuracy.dist.lda,accuracy.dist.qda,accuracy.dist.gbm,
                          accuracy.dist.rpart,accuracy.dist.treebag,
                          accuracy.dist.rf)

colnames(accuracy.dists.all)<-c("LDA","QDA","GBM","RPART","TREEBAG","RF")

library(reshape2)

accuracy.dists.all.melted<-melt(accuracy.dists.all)

colnames(accuracy.dists.all.melted)[2]<-c("Model")

library(ggplot2)

ggplot(accuracy.dists.all.melted,aes(x=Model,y=value)) +
                    geom_violin(scale = "width",aes(fill=Model)) +
                    xlab("Model Type") +
                    ylab("Accuracy") +
                    ggtitle("Out of Sample Error Estimation")
            

Accuracy.Mean<-tapply(accuracy.dists.all.melted$value,
                INDEX=list(accuracy.dists.all.melted$Model),mean)

Accuracy.SD<-tapply(accuracy.dists.all.melted$value,
              INDEX=list(accuracy.dists.all.melted$Model),sd)

accuracy.summary<-round(rbind(Accuracy.Mean,Accuracy.SD),4)




##Try bagged nominal logistic regression.QDA/LDA, random forest, and adaboost. 

##1)Train all models on training.1
##2)Predict all models on training.2 
##3)Use a decision tree to stack the predictions from the training 2 set.
##4)Use the new stacked model to predict on test.
