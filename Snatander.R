setwd("C:\\Users\\kumarvch\\Desktop\\Data_aNALYSIS\\ds Project\\santander Prject")

train = read.csv("train.csv")
head(train)

test = read.csv("test(1).csv")
head(test)


######Missing value analysis##########

Missing_value = data.frame(apply(train,2 , function(x){sum(is.null(x))}))
Missing_value

#Found no missing values in the data

##############Outlier Ananlysis############

Numeric_data = train[ , -1:-2]
View(Numeric_data)

cnames = colnames(Numeric_data)

for (i in cnames){
  val = train[,i][train[,i] %in% boxplot.stats(train[,i])$out]
  print(length(val))
  train[,i][train[,i] %in% val] = NA
}

#we have found outliers in the data and we replaced all the outliers with NA

#lets impute them with knn imputation

#install.packages("DMwR")

#library(DMwR)

#train= knnImputation(train, k=3)

#Knn imputation is taking lot of time so we impute by Median 

for(i in cnames){
  train[,i][is.na(train[,i])] = median(train[,i], na.rm = T)
}

#Now we have imputed all the missing values, Lets cross check 

missing_v = data.frame(apply(train,2 , function(x){sum(is.null(x))}))
missing_v

############# Feature selection ############

train1 = train

levels(factor(train1$target))

str(train1$target)

#lets conver the target feature into factor.

train1$target = as.factor(train1$target)

str(train1$target)

levels(factor(train1$target))


table(train1$target)

(table(train$target)/length(train$target))*100

install.packages("ggplot2")

library(ggplot2)

plot = ggplot(train,aes(target))+theme_bw()+geom_bar(stat='count',fill='red3')
plot

#clearly the dataset is imbalance.

###Principle component analysis

target = train1$target
data  = train1[,-1:-2]
View(data)
pca = princomp(na.omit(data),cor = F)
summary(pca)

plot(pca)

gof = (pca$sdev)^2/sum((pca$sdev)^2)
sum(gof[1:131])

plot(gof, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b")

plot(cumsum(gof), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")

newdata = pca$scores[,1:131]

newdata = cbind(target,newdata)

View(newdata)

colnames(newdata) = c("target","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15","p16","p17","p18","p19","p20","p21","p22","p23","p24",
                      "p25","p26","p27","p28","p29","p30","p31","p32","p33","p34","p35","p36","p37","p38","p39","p40","p41","p42","p43","p44","p45","p46",
                      "p47","p48","p49","p50","p51","p52","p53","p54","p55","p56","p57","p58","p59","p60","p61","p62","p63","p64","p65","p66","p67","p68",
                      "p69","p70","p71","p72","p73","p74","p75","p76","p77","p78","p79","p80","p81","p82","p83","p84","p85","p86","p87","p88","p89","p90",
                      "p91","p92","p93","p94","p95","p96","p97","p98","p99","p100","p101","p102","p103","p104","p105","p106","p107","p108","p109","p110",
                      "p111","p112","p113","p114","p115","p116","p117","p118","p119","p120","p121","p122","p123","p124","p125","p126","p127","p128","p129","p130","p131")

newdata=as.data.frame(newdata)

head(newdata)

newdata$target = as.factor(newdata$target)

View(newdata)

newdata$target = ifelse(newdata$target == 1 , 0 ,1 )

View(newdata)

##########Modeling#########

# Test train split

install.packages("caret")

library(caret)

part = createDataPartition(newdata$target , times = 1 ,p = 0.7)

train = newdata[part$Resample1,]
test = newdata[-part$Resample1, ]

#Logistic regression

install.packages("glmnet")
install.packages("pROC")

logit = glm(target~., data = train ,family = "binomial")


logit_pred = predict(logit,newdata = test ,type = "response")

logit_pred = ifelse(logit_pred >0.5 , 1 ,0)

Conf_Mat = table(test$target , logit_pred)
Conf_Mat

#      0     1
# 0 53468   446
# 1  5227   859

#Accuracy = 90.54% 

(859+53468)/(53468+446+5227+859)

#precision = 0.65%

(859)/(446+859)

#recall score or sensitivity = 14.14

(859)/(5227+859)

# specificity = 99.12
(53468)/(53468+446)

#Our business objective is to optimise recall score becouse 
#we cannot loose a customer who will make transaction in the future
#so we are reducing the thresh hold value

logit2 = glm(target~., data = train ,family = "binomial")


logit_pred2 = predict(logit2,newdata = test ,type = "response")

#here we are reducing the thresh hold value to 0.2
#i.e predictions falling above 0.2 will be assighned as 1 and falling below 0.2 will be assighned to 0

logit_pred2 = ifelse(logit_pred2 >0.12 , 1 ,0)

Conf_Mat2 = table(test$target , logit_pred2)
Conf_Mat2

#    0     1
#0 42455  11459
#1  2009  4077

#recall score = 66.98

4077/(2009+4077)

# Accuracy = 77.5%

(42455+4077)/(42455+4077+2009+11459)

#by dicreasing  the thresh Hold value to 0.12, recall score has 
#been increased to ~67% at the cost of reducing the accuracy to 77.5%

#ROC curve

library(pROC)


plot(roc(test$target,logit_pred, direction="<"))







