# 1. Install packages to read the NCBI's GEO microarray SOFT files in R
# 1.Ref. http://www2.warwick.ac.uk/fac/sci/moac/people/students/peter_cock/r/geo/

# 1.1. Uncomment only once to install stuff

#source("https://bioconductor.org/biocLite.R")
#biocLite("GEOquery")
#biocLite("Affyhgu133aExpr")


# 1.2. Use packages # Comment to save time after first run of the program in an R session

library(Biobase)
library(GEOquery)

# Add other libraries that you might need below this line



# 2. Read data and convert to dataframe. Comment to save time after first run of the program in an R session
# 2.1. Once download data from ftp://ftp.ncbi.nlm.nih.gov/geo/datasets/GDS2nnn/GDS2771/soft/GDS2771.soft.gz
# 2.Ref.1. About data: http://www.ncbi.nlm.nih.gov/sites/GDSbrowser?acc=GDS2771
# 2.Ref.2. Study that uses that data http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3694402/pdf/nihms471724.pdf
# 2.Warning. Note that do not use FULL SOFT, only SOFT, as mentioned in the link above. 2.2.R. http://stackoverflow.com/questions/20174284/error-in-gzfilefname-open-rt-invalid-description-argument

gds2771 <- getGEO(filename='/home/sam/Desktop/sem-7/EE622/Assignment/GDS2771.soft.gz') # Make sure path is correct as per your working folder. Could be './GDS2771.soft.gz'
eset2771 <- GDS2eSet(gds2771) # See http://www2.warwick.ac.uk/fac/sci/moac/people/students/peter_cock/r/geo/

# 2.2. View data (optional; can be commented). See http://www2.warwick.ac.uk/fac/sci/moac/people/students/peter_cock/r/geo/
eset2771 # View some meta data
featureNames(eset2771)[1:10] # View first feature names
sampleNames(eset2771) # View patient IDs. Should be 192
pData(eset2771)$disease.state #View disease state of each patient. Should be 192

# 2.3. Convert to data frame by concatenating disease.state with data, using first row as column names, and deleting first row
data2771 <- cbind2(c('disease.state',pData(eset2771)$disease.state),t(Table(gds2771)[,2:194]))
colnames(data2771) = data2771[1, ] # the first row will be the header
data2771 = data2771[-1, ] 

# 2.4. View data frame (optional; can be commented)
#View(data2771)



# WRITE YOUR CODE BELOW THIS LINE
#Number of rows and columns in given data
nrow(data2771)
ncol(data2771)

#removing all SUSPECT CANCER and NA values
data2771<-data2771[1:187,1:(ncol(data2771)-68)]
sum(is.na(data2771))

#Converting the given data to numeric matrix
is.numeric(data2771)
data_num<-matrix(as.numeric(unlist(data2771)),nrow=nrow(data2771))
nrow(data_num)
ncol(data_num)
sum(is.na(data_num))
is.numeric(data_num)
#setting seed for generating pseudo random numbers
set.seed(7)

#Using multiple cores of CPU for faster processing
library(doMC)
registerDoMC(cores = 4)

#Using different models for predictions
library(glmnet)

#Building a cross-validated model for Ridge
ridge = cv.glmnet(data_num[,2:ncol(data_num)], data_num[,1], family = "binomial", type.measure = "class",alpha=0)
plot(ridge)
#Lamda value for best accuracy obtained which was tested on 100 random values
ridge$lambda.min
#lamda for least variant validation accuracy
ridge$lambda.1se
#Best accuracy
print(best_accu_ridge<-(1-min(ridge$cvm))*100)
#The weights(beta) of features for the best model 
w_ridge<-(coef(ridge, s = ridge$lambda.min))
sum(w_ridge!=0)
plot(w_ridge)
#Effect on co-efficients with changing lamda
ridge_fit = glmnet(data_num[,2:ncol(data_num)], data_num[,1], family = "binomial",alpha=0)
plot(ridge_fit,xvar="lambda")

#Building a cross-validated model for Lasso
lasso = cv.glmnet(data_num[,2:ncol(data_num)], data_num[,1], family = "binomial", type.measure = "class",alpha=1)
plot(lasso)
#Lamda value for best accuracy obtained which was tested on 100 random values
lasso$lambda.min
#lamda for least variant validation accuracy
lasso$lambda.1se
#Best accuracy
print(best_accu_lasso<-(1-min(lasso$cvm))*100)
#The weights(beta) of features for the best model 
w_lasso<-(coef(lasso, s = lasso$lambda.min))
sum(w_lasso!=0)
plot(w_lasso)
#Effect on co-efficients with changing lamda
lasso_fit = glmnet(data_num[,2:ncol(data_num)], data_num[,1],alpha=1,family = "binomial")
plot(lasso_fit,xvar="lambda")

#Building a cross-validated model for Elastic net with alpha=0.3
elastic = cv.glmnet(data_num[,2:ncol(data_num)], data_num[,1], alpha = 0.3,family = "binomial", type.measure = "class")
plot(elastic)
#Lamda value for best accuracy obtained which was tested on 100 random values
elastic$lambda.min
#lamda for least variant validation accuracy
elastic$lambda.1se
#Best accuracy
print(best_accu_elastic<-(1-min(elastic$cvm))*100)
#The weights(beta) of features for the best model 
w_elastic = coef(elastic, s = elastic$lambda.min, exact = FALSE)
plot(w_elastic)
sum(w_elastic!=0)
#Effect on co-efficients with changing lamda
elastic_fit = glmnet(data_num[,2:ncol(data_num)], data_num[,1], family = "binomial",alpha=0.3)
plot(elastic_fit,xvar="lambda")
#summary(coef.apprx)

library(caret)

#Converting target values to levels
y=data_num[,1]
y[y==1]<-"Yes"
y[y==2]<-"No"

#Training a model with all possible combinations of alpha nad lamda
glmnet_grid <- expand.grid(alpha = c(0, .2, .4, .5, .6, .8, 1),
                           lambda = seq(.01, 1, length = 20))
glmnet_ctrl <- trainControl(method = "cv", number = 10)
glmnet_fit <- train(data_num[,2:ncol(data_num)],y,
                    method = "glmnet",
                    tuneGrid = glmnet_grid,
                    trControl = glmnet_ctrl,family="binomial",metric ="Accuracy")

plot(glmnet_fit)
#Alpha and lambda values for best accuracy obtained
glmnet_fit$bestTune
glmnet_fit
#the top 20% most important variables
varImp(glmnet_fit)

library("e1071")
#SVM
svm_model<- svm(x=data_num[,2:ncol(data_num)],y=as.factor(data_num[,1]),cross=15)
plot(svm_model$accuracies)
#Average accuracy across all validations
svm_model$tot.accuracy
summary(svm_model)

#Most important genes checking by value of co-efficients
colnames(data2771)[which(coef(elastic)>0.01)]
colnames(data2771)[which(coef(lasso)>0.01)]
colnames(data2771)[which(coef(ridge)>0.005)]
