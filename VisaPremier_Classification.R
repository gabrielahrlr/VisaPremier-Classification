##############################################################################################
# Gabriela HERNÁNDEZ & Maìra LADEIRA
# ASSIGNMENT Model Based  Clustering and Classification 
#  Prof. Julien JACQUES
##############################################################################################

# Libraries
library(MASS) 
library(mclust)
library(Rmixmod)
library(caret)
library(pROC)


# Fixing the continous data
data_visa=read.table("~/Documents/DMKM-3/Numerical-Learning/Julien-Jacques/VisaPremier.txt",header=TRUE,na.strings =c(".", "F."))
summary(data_visa)
# Index of categorical features
ind_categ_feature=c(1,2,3,4,6,8,9,45,46 ,47)
# Extraction of only the continuous features
data_continuous=data_visa[,-ind_categ_feature]

# Automatically replace the NA with the mean of each feature
summary(data_continuous)
for (j in 1:ncol(data_continuous)){
  if (sum(is.na(data_continuous[,j]))>0) data_continuous[is.na(data_continuous[,j]),j]=mean(data_continuous[,j],na.rm=TRUE)
}
summary(data_continuous)
# Remove the variable nbimpaye which is constant
data_continuous$nbimpaye=NULL
data_continuous <- data_continuous[-2,]
#data_visa <- data_visa[-2,]
data_continuous$mtbon=NULL
data_continuous$nbbon=NULL


for (j in 1:ncol(data_continuous)){
  data_continuous[,j] <- as.numeric(data_continuous[,j])
}

# Fixing categorical features
# extraction of only the categ. features
data_categ=data_visa[,ind_categ_feature]
# A look on the categorical feature
summary(data_categ)
# There are  NA's in departem and codeqlt) and some features are registred as integer
# Replace NA's with the mode
tmp=sort(table(data_categ$departem),decreasing=TRUE)
data_categ$departem[is.na(data_categ$departem)]=as.integer(names(tmp)[1])
# Convert department to factor
data_categ$departem=factor(data_categ$departem)
# Replace NA's with the mode
data_categ$codeqlt=as.character(data_categ$codeqlt)
tmp=sort(table(data_categ$codeqlt),decreasing=TRUE)
data_categ$codeqlt[is.na(data_categ$codeqlt)]=names(tmp)[1]
data_categ$codeqlt <- as.factor(data_categ$codeqlt)
# Replace NA's with the mode
data_categ$sitfamil=as.character(data_categ$sitfamil)
tmp=sort(table(data_categ$sitfamil),decreasing=TRUE)
data_categ$sitfamil[is.na(data_categ$sitfamil)]=names(tmp)[1]
data_categ$sitfamil <- as.factor(data_categ$sitfamil)
# Transform matricul and ptvente into factor
data_categ$matricul=factor(data_categ$matricul) 
data_categ$ptvente=factor(data_categ$ptvente)
# Remove sexer and cartevpr, these are duplicates of sexe and cartevp
data_categ$sexer=NULL
data_categ$cartevpr=NULL
# Remove Customers ID
data_categ$matricul=NULL
# Record 2 is an outlier, thus we remove it
data_categ=data_categ[-2,]
# View again in the data_categ
summary(data_categ)

# Data Spliting
# Set seed to be able to reproduce results
set.seed(2016)
n <- nrow(data_continuous)
trainRows <- as.integer(0.7*n)
s <- sample(1:n, trainRows, replace=F)

# Partition of the train dataset into predictor  continuous variables and class
# Save predictor variables in data.train
data.train <- data_continuous[s,]

#data.train <- scale(data.train, center=FALSE, scale = TRUE)
# Save class (variable to predict) in data.class
train.class <- data_categ[s,"cartevp"]

# Partition of the test dataset into predictor continuous variables and class
# Save predictor variables in data.test
data.test <- data_continuous[-s,]

# Save class (variable to predict) in data.class.test
test.class <- data_categ[-s,"cartevp"]

########################################################################
#              LDA model
########################################################################
lda.model <- MclustDA(data.train, train.class, modelType = "EDDA", modelNames = "EEE")
# Check the misclassification error in the train set
summary(lda.model, parameters = TRUE)
# Check the misclassification error in the test set
summary(lda.model, newdata = data.test, newclass = test.class)
# Predict again with the test set to get statistics
lda.prediction <- predict(lda.model, newdata=data.test,  newclass=data.class.test)
# Get the confusion Matrix and Statistics
confusionMatrix( lda.prediction$classification, test.class)
# Plot ROC Curve
lda.ROC <- roc(test.class, lda.prediction$z[, "Cnon"])
plot(lda.ROC, type = "S", col='green', main='ROC Curve', print.thres=0.5)
legend("bottomright", "LDA", cex = 0.3, lty=1, col="green")

########################################################################
#              QDA Model
########################################################################
# We first need to jitter the data to avoid exact multicolinearity
data.train.J <- data.train
data.train.J[, -1] <- apply(data.train[, -1], 2, jitter)
# Build the QDA prediction model with the the jitter data
qda.model <- qda(train.class~., data.train.J)
# Prediction on the test set
qda.prediction <- predict(qda.model, newdata=data.test,  newclass=data.class.test)
# Confusion Matrix
confusionMatrix(qda.prediction$class, test.class)
# ROC Curve
qda.ROC <- roc(test.class, qda.prediction$posterior[, "Cnon"])
plot(qda.ROC, type = "S", col='blue', main='ROC Curve', print.thres=0.5)
legend("bottomright", "QDA", cex = 0.3, lty=1, col="blue")

########################################################################
#              SVM Radial Model
########################################################################
# Set control parameters:
# 10-fold cross validation with three repetitions
cvCtrl <- trainControl(method = "repeatedcv", repeats = 3,
                       summaryFunction = twoClassSummary,
                       classProbs = TRUE)
set.seed(1)
svmModel <- train(x=data.train, y=train.class, method="svmRadial",
                  metric="ROC", trControl=cvCtrl)
# Check the parameters of the final model
svmModel$finalModel
# Make the predictions on the test set
svmPredict <- predict(svmModel, data.test)
svmPredict.Prob <- predict(svmModel, data.test, type="prob")
# Confusion Matrix
confusionMatrix(svmPredict, test.class)
# ROC Curve
svm.ROC <- roc(test.class, svmPredict.Prob[, "Cnon"])
plot(svm.ROC, type = "S", col='pink', main='ROC Curve', print.thres=0.5)
legend("bottomright", "SVM-Radial", cex = 0.3, lty=1, col="pink")

########################################################################
# Model based on Continuous and categorical features: Mixture Model
########################################################################
# Joining datasets
X <- cbind(data_continuous, data_categ)
class <- c("cartevp")
class_rem <- which(names(X) %in% class)
# Removing the class from the training set
X.train <- X[s,-c(class_rem)]
X.train.class <- X[s,'cartevp']
# Preprocess the continuous variables by doing center, scale and pca
data.preProc.cont <- preProcess(X.train[,-c(36,37,38,39,40,41)], method=c("center","scale","pca"), outcome = X.train.class)
X.train.scaled <- predict(data.preProc.cont, X.train)
# Removing the class from the test set
X.test <- X[-s, -c(class_rem)]
# Applying pca transformation to the test set
X.test.scaled <- predict(data.preProc.cont, X.test)
X.test.class <- X[-s, 'cartevp']

# Mix Model
# For  Learning the model using heterogeneous data, dataType="composite" must be specified
# the criterion to choose the best model is CV.
mix.learn <- mixmodLearn(X.train.scaled, knownLabels = X.train.class, criterion = "CV",dataType = "composite")
# Predict with the new data
mix.prediction <- mixmodPredict(X.test.scaled, mix.learn["bestResult"])
# Confusion Matrix
confusionMatrix(mix.prediction["partition"], as.factor(as.numeric(X.test.class)))
par(pty="s")
mix.ROC <- roc(X.test.class, mix.prediction@proba[,2])
plot(mix.ROC, type='S',col='black', main='ROC Curve', print.thres=0.5)
legend("bottomright", "Mixture Model", cex=0.3 , lty=1, col="black")

###############################################################################
#
#Comparison among models
###############################################################################

# Plot LDA ROC Curve
lda.ROC <- roc(test.class, lda.prediction$z[, "Cnon"])
plot(lda.ROC, type = "S", col='green')
par(new=T)
#legend("bottomright", "LDA", cex = 0.3, lty=1, col="green")

# Plot QDA ROC Curve
qda.ROC <- roc(test.class, qda.prediction$posterior[, "Cnon"])
plot(qda.ROC, type = "S", col='blue')
#legend("bottomright", "QDA", cex = 0.3, lty=1, col="blue")
par(new=T)
# Plot SVM ROC Curve
svm.ROC <- roc(test.class, svmPredict.Prob[, "Cnon"])
plot(svm.ROC, type = "S", col='pink')
#legend("bottomright", "SVM-Radial", cex = 0.3, lty=1, col="pink")
par(new=T)
# Plot Mixmod ROC Curve
mix.ROC <- roc(X.test.class, mix.prediction@proba[,2])
plot(mix.ROC, type='S',col='black', main='ROC Curves')
par(cex=0.7)
legend("bottomright", c("LDA","QDA","SVM","Mixture Model"), cex=0.8 , lty=c(1,1,1,1), col=c("green",'blue','pink','black'))
