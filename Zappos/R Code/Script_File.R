#This file contains the code for the Zappos Advanced Analytics Summer Internship 2015.
#Author : Anurag Ladage.

##Libraries Used
library(corrgram)
library(Amelia)
library(rattle)
library(ROCR)
library(nnet)
library(plyr)
library(reshape)
library(ggplot2)
library(wle)
library(sandwich)
library(pscl)
library(lmtest)

##User-defined Functions Used
### imputeByFactor : Function which takes in arguments a dataset,column to be imputed and a factor column. The values in the column_impute are averaged for each factor and the missing values are replaced by one of these average values  depending on which factor they correspond to.
imputeByFactor <- function(data,column_impute, column_factor)
{
    averagetable <- tapply(data[,column_impute],as.factor(data[,column_factor]),mean,na.rm=TRUE)
    
    for(i in 1:nrow(data))
    {
        if(is.na(data[i,column_impute]))
        {
            data[i,column_impute] = averagetable[which(row.names(averagetable)==data[i,column_factor])]
        }
    }
    return(data)
}

### multiplot: This function is copied from The cookbook for R website entirely. It draws multiple plots in the same figure. The code for both the functions is at the end of this document.
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
    require(grid)
    
    # Make a list from the ... arguments and plotlist
    plots <- c(list(...), plotlist)
    
    numPlots = length(plots)
    
    # If layout is NULL, then use 'cols' to determine layout
    if (is.null(layout)) {
        # Make the panel
        # ncol: Number of columns of plots
        # nrow: Number of rows needed, calculated from # of cols
        layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                         ncol = cols, nrow = ceiling(numPlots/cols))
    }
    
    if (numPlots==1) {
        print(plots[[1]])
        
    } else {
        # Set up the page
        grid.newpage()
        pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
        
        # Make each plot, in the correct location
        for (i in 1:numPlots) {
            # Get the i,j matrix positions of the regions that contain this subplot
            matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
            
            print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                            layout.pos.col = matchidx$col))
        }
    }
}

##The dataset is assumed to be in the working directory.
##Working directory can be obtained in R by using getwd() and can be set by setwd().
my_data <- read.csv("Analytics Challenge Data 2.csv",header=T)
my_data <- my_data[,1:12]
my_data$day <- as.factor(as.Date(gsub(pattern=" 0:00",x=as.character(my_data$day),replacement=""),format="%m/%d/%Y"))
my_data$new_customer <- as.factor(my_data$new_customer)
my_data$platform <- as.character(my_data$platform)
blank_char <- unique(my_data$platform)[10]
for(i in which(my_data$platform==blank_char | my_data$platform=="Unknown"))
{
    my_data$platform[i]=NA
}
my_data$platform <- as.factor(my_data$platform)
head(my_data)

##Excluding the missing data
na.exclude(my_data)
#The amount of missing data which gets excluded is almost half of the total data available. Thus it is not advisable to waste that much data. Thus we look to impute the missing data before diving into our analysis.

##Imputing missing data
windows(height=7)
missmap(my_data,main="Missing Data Plot")
##From the above plot we can see there is missing data only in columns named "new_customer" and "gross_sales". At this point it is important to understand that garbage data will led to garbage analysis. Thus great importance should be given to data imputation as they can lead to very erroneous results. First we will impute the gross_sales column and then use machine learning to impute "new_customer" column.
imputed_data_byDay <- imputeByFactor(my_data,8,1)
imputed_data_bySite <- imputeByFactor(my_data,8,2)
mean_table <- c(mean(my_data$gross_sales,na.rm=T),mean(imputed_data_byDay$gross_sales),mean(imputed_data_bySite$gross_sales))
names(mean_table) <- c("original_mean","mean_byDay","mean_bySite")
mean_table
## The mean_table list the mean of the gross_sales from the original data and from the imputed tables. We observe that when we impute the data using day as the imputing factor the mean is closest to that of the original. Thus, imputed_data_byDay the dataset we need.
my_data <- imputed_data_byDay

##Imputation for the Platform factor variable.
rattle()
missmap(my_data)
my_data_platform_train <- my_data[which(!is.na(my_data$platform)),]
my_data_platform_train_train <- my_data_platform_train[1:15000,]
my_data_platform_train_test <- my_data_platform_train[15000:nrow(my_data_platform_train),]
my_data_platform_test <- my_data[which(is.na(my_data$platform)),]
my_data <- my_data_platform_train

##Now we try to predict the missing data in the "new_customer" column.
my_data_ml <- na.exclude(my_data)
my_data_predict <- my_data[is.na(my_data$new_customer),]
##We use the rattle package to explore quickly which techique can be used for prediction of the "new_customer" data. After looking at couple of different techniques we reach the conclusion that neural network gives the least error rate on the test data set.
##The my_data_ml dataset defined above is the rows with "new_customer" values present in it. The my_data_predict is the dataset with new_customer values absent in it. We use the my_data_ml dataset to train our neural network and then from that predict the missing values in my_data_predict.
nn_my_data_train <- my_data_ml[1:8500,]
nn_my_data_test <- my_data_ml[8501:nrow(my_data_ml),]
nn_my_data_ytrain <- nn_my_data_train$new_customer
nn_my_data_xtrain <- subset(nn_my_data_train,select= -c(new_customer))
nn_my_data_ytest <- nn_my_data_test$new_customer
nn_my_data_xtest <- subset(nn_my_data_test,select= -c(new_customer))
set.seed(199)
my_data_nnet <- nnet(as.factor(new_customer) ~ .,data=nn_my_data_train, size=30, skip=TRUE, MaxNWts=10000, trace=T, maxit=100)
nn_my_data_yhat <- predict(my_data_nnet,nn_my_data_xtest,type='class')
table(nn_my_data_ytest,nn_my_data_yhat)



my_data_predict$new_customer <- predict(my_data_nnet,subset(my_data_predict,select=-c(new_customer)),type="class")
my_data <- rbind(my_data_ml,my_data_predict)
my_data$new_customer <- as.factor(my_data$new_customer)
##At this point my_data is the dataset is a clean dataset with no missing values.



##Generating general summary and correlation plots for the data
###install.packages("corrgram") <- If the package is not installed.
windows()
par(bg="grey")
corrgram(my_data,lower.panel=panel.bar,upper.panel=panel.pie,diag.panel=panel.minmax, text.panel=panel.txt, main="Correlation Matrix")
str(my_data)

##Calculating the required matrices.
my_data$conversion_rate <- my_data$orders/my_data$visits
my_data$conversion_rate <- as.numeric(revalue(as.character(my_data$conversion_rate),replace=c("NaN"=0))) 
my_data$bounce_rate <- my_data$bounces/my_data$visits
my_data$bounce_rate <- as.numeric(revalue(as.character(my_data$bounce_rate),replace=c("NaN"=0))) 
my_data$add_to_cart_rate <- my_data$add_to_cart/my_data$visits 
my_data$add_to_cart_rate <- as.numeric(revalue(as.character(my_data$add_to_cart_rate),replace=c("NaN"=0))) 
requested_metrices <- my_data[,13:15]
write.csv(my_data, file="Final_Data_Set.csv")

##Making Different Plots
###Plot 1
windows()
plot1_data <- my_data[,c(1,3,13,14,15)]
plot1_data$new_customer <- as.factor(plot1_data$new_customer)
plot1_data <- ddply(plot1_data,.(day,new_customer),summarize,avg_conversion_rate = mean(conversion_rate),avg_bounce_rate = mean(bounce_rate),avg_add_to_cart_rate=mean(add_to_cart_rate))
plot1_conversion_rate <- qplot(x=day,y=avg_conversion_rate,group=new_customer,data=plot1_data,col=new_customer,main="Average Conversion Rate Graph (Plot 1)",xlab="Day", ylab="Daily Avg Conversion Rate")
plot1_conversion_rate <- plot1_conversion_rate + geom_line()
plot1_conversion_rate <- plot1_conversion_rate + scale_x_discrete(breaks=levels(plot1_data$day)[seq(1,268,14)],labels=levels(plot1_data$day)[seq(1,268,14)]) + theme(axis.text.x = element_text(angle = 90, hjust = 1))
plot1_bounce_rate <- qplot(x=day,y=avg_bounce_rate,group=new_customer,data=plot1_data,col=new_customer,main="Average Bounce Rate Graph (Plot 1)",xlab="Day", ylab="Daily Avg Bounce Rate")
plot1_bounce_rate <- plot1_bounce_rate + geom_line()
plot1_bounce_rate <- plot1_bounce_rate + scale_x_discrete(breaks=levels(plot1_data$day)[seq(1,268,14)],labels=levels(plot1_data$day)[seq(1,268,14)]) + theme(axis.text.x = element_text(angle = 90, hjust = 1))
plot1_add_to_cart_rate <- qplot(x=day,y=avg_add_to_cart_rate,group=new_customer,data=plot1_data,col=new_customer,main="Average Add to Cart Rate Graph (Plot 1)",xlab="Day", ylab="Daily Avg Add To Cart Rate")
plot1_add_to_cart_rate <- plot1_add_to_cart_rate + geom_line()
plot1_add_to_cart_rate <- plot1_add_to_cart_rate + scale_x_discrete(breaks=levels(plot1_data$day)[seq(1,268,14)],labels=levels(plot1_data$day)[seq(1,268,14)]) + theme(axis.text.x = element_text(angle = 90, hjust = 1))
plot1_conversion_rate
plot1_bounce_rate
plot1_add_to_cart_rate
multiplot(plot1_conversion_rate,plot1_bounce_rate,plot1_add_to_cart_rate)

###Plot 2
windows()
plot2_data <- my_data[,c(1,2,13,14,15)]
plot2_data <- ddply(plot2_data,.(day,site),summarize,avg_conversion_rate = mean(conversion_rate),avg_bounce_rate = mean(bounce_rate),avg_add_to_cart_rate=mean(add_to_cart_rate))
plot2_conversion_rate <- qplot(x=day,y=avg_conversion_rate,ylim=c(0,1),group=site,data=plot2_data,col=site,main="Average Conversion Rate Graph (Plot 2)",xlab="day", ylab="Avg Conversion Rate")
plot2_conversion_rate <- plot2_conversion_rate + geom_line()
plot2_conversion_rate <- plot2_conversion_rate + scale_x_discrete(breaks=levels(plot2_data$day)[seq(1,268,14)],labels=levels(plot2_data$day)[seq(1,268,14)]) + theme(axis.text.x = element_text(angle = 90, hjust = 1))
plot2_bounce_rate <- qplot(x=day,y=avg_bounce_rate,group=site,data=plot2_data,col=site,main="Average Bounce Rate Graph (Plot 2)",xlab="day", ylab="Avg Bounce Rate")
plot2_bounce_rate <- plot2_bounce_rate + geom_line()
plot2_bounce_rate <- plot2_bounce_rate + scale_x_discrete(breaks=levels(plot2_data$day)[seq(1,268,14)],labels=levels(plot2_data$day)[seq(1,268,14)]) + theme(axis.text.x = element_text(angle = 90, hjust = 1))
plot2_add_to_cart_rate <- qplot(x=day,y=avg_add_to_cart_rate,group=site,data=plot2_data,col=site,main="Average Add to Cart Rate Graph (Plot 2)",xlab="day", ylab="Avg Add To Cart Rate")
plot2_add_to_cart_rate <- plot2_add_to_cart_rate + geom_line()
plot2_add_to_cart_rate <- plot2_add_to_cart_rate + scale_x_discrete(breaks=levels(plot2_data$day)[seq(1,268,14)],labels=levels(plot2_data$day)[seq(1,268,14)]) + theme(axis.text.x = element_text(angle = 90, hjust = 1))
multiplot(plot2_conversion_rate,plot2_bounce_rate,plot2_add_to_cart_rate)
plot2_conversion_rate

###Plot 3
windows()
plot3_data <- my_data[,c(1,4,13,14,15)]
plot3_data <- ddply(plot3_data,.(day,platform),summarize,avg_conversion_rate = mean(conversion_rate),avg_bounce_rate = mean(bounce_rate),avg_add_to_cart_rate=mean(add_to_cart_rate))
plot3_conversion_rate <- qplot(x=day,y=avg_conversion_rate,group=platform,data=plot3_data,col=platform,main="Average Conversion Rate Graph (Plot 2)",xlab="day", ylab="Avg Conversion Rate")
plot3_conversion_rate <- plot3_conversion_rate + geom_line()
plot3_conversion_rate <- plot3_conversion_rate + scale_x_discrete(breaks=levels(plot3_data$day)[seq(1,268,14)],labels=levels(plot3_data$day)[seq(1,268,14)]) + theme(axis.text.x = element_text(angle = 90, hjust = 1))
plot3_bounce_rate <- qplot(x=day,y=avg_bounce_rate,group=platform,data=plot3_data,col=platform,main="Average Bounce Rate Graph (Plot 2)",xlab="day", ylab="Avg Bounce Rate")
plot3_bounce_rate <- plot3_bounce_rate + geom_line()
plot3_bounce_rate <- plot3_bounce_rate + scale_x_discrete(breaks=levels(plot3_data$day)[seq(1,268,14)],labels=levels(plot3_data$day)[seq(1,268,14)]) + theme(axis.text.x = element_text(angle = 90, hjust = 1))
plot3_add_to_cart_rate <- qplot(x=day,y=avg_add_to_cart_rate,group=platform,data=plot3_data,col=platform,main="Average Add to Cart Rate Graph (Plot 2)",xlab="day", ylab="Avg Add To Cart Rate")
plot3_add_to_cart_rate <- plot3_add_to_cart_rate + geom_line()
plot3_add_to_cart_rate <- plot3_add_to_cart_rate + scale_x_discrete(breaks=levels(plot3_data$day)[seq(1,268,14)],labels=levels(plot3_data$day)[seq(1,268,14)]) + theme(axis.text.x = element_text(angle = 90, hjust = 1))
multiplot(plot3_conversion_rate,plot3_bounce_rate,plot3_add_to_cart_rate)
plot3_conversion_rate


##Regression Analysis
###Poisson Regression
###In my regression analysis I have assumed that the variable of interest is orders. Thus we take orders as the response variable and rest as the input variables. Also I have not excluded some variables either due to their redundancy in the regression model or due to them being not important from the point of view of our analysis. Also it is important to keep in mind that simple logistics regression should be applied as our response variable is not continuous but an integer/count variable. Thus, keeping this in mind I have used poisson regression for the purpose of this analysis.
my_data_poisson <- my_data[,c(3,5,7,9,10,11,12)]
#The anova test both eliminate the product_page_views variable.
poisson_fit1 <- glm(orders~.,family="poisson",data=my_data_poisson)
poisson_fit2 <- glm(orders~.-product_page_views,family="poisson",data=my_data_poisson)
poisson_fit3 <- glm(orders~.-product_page_views-bounces,family="poisson",data=my_data_poisson)
anova(poisson_fit1,poisson_fit2,poisson_fit3)
step(poisson_fit3)
#The anova test both eliminate the product_page_views variable.
final_model <-poisson_fit3
model_coefficients <- coefficients(poisson_fit3)
model_coefficients
confint(final_model)




