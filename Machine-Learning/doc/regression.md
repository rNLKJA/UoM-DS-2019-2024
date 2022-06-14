### Regression Analysis

Regression analysis is a statistical method to model the relationship between a dependent (target) and independent (predictor) variables with one or more independent variables. More specifically, Regression analysis helps us to understand how the value of the dependent variable is changing corresponding to an independent variable when other independent variables are held fixed. It predicts continuous/real values such as temperature, age, salary, price, etc.

Regression is a supervised learning technique, which helps in finding the correlation between variables and enables us to predict the continuous output variable based on the one or more predictor variables. It is mainly used for prediction, forecasting, time series modeling, and determining the causal-effect relationship between variables.

In regression, we plot a graph between the variables which best fits the given datapoints, using this plot, the machine learning model can make predictions about the data. In simple words:

> Regression shows a line or curve that passes through all the datapoints on target-predictor graph in such a way that the vertical distance between the datapoints and the regression line is minimum.

The distance between datapoints and line tells whether a model has captured a strong relationship or not. Some examples of regression can be as:
- Prediction of rain using temperature and other factors
- Determining Market trends
- Prediction of road accidents due to rash driving

### Terminologies Related to the Regression Analysis
- *Dependent Variable*: The main factor in Regression analysis which we want to predict or understand is called the dependent variable. It is also called target variable.
- *Independent Variable*: The factors which affect the dependent variables or which are used to predict the values of the dependent variables are called independent variable, also called predictor.
- *Outliers*: Outlier is an observation which contains either very low value or very high value in comparison to other observed values. An outlier may hamper the result, so it should be avoided.
- *Multicollinearity*: If the independent variables are highly correlated with each other than other variables, then such condition is called Multicollinearity. It should not be present in the dataset, because it creates problem while ranking the most affecting variable.
- *Underfitting and Overfitting*: If our algorithm works well with the training dataset but not well with test dataset, then such problem is called Overfitting. And if our algorithm does not perform well even with training dataset, then such problem is called underfitting.

### Reason to use Regression Analysis
As mentioned above, Regression analysis helps in prediction of a continuous variable. There are various scenarios in the real world where we need some future predictions such as weather condition, sales prediction, marketing trends, etc., for such case we need some technology which can make predictions more accurately. So for such case we need Regression analysis which is a statistical method and used in machine learning and data science. Below are some other reasons for using Regression analysis:
- Regression estimates the relationship between the target and the indepedent variable.
- It is used to find the trends in data.
- It helps to predict real/continous values.
- By performing the regression, we can confidently determine the most important factor, the least import factor, and how each factor is affecting the other factors.

### Types of Regression
There are various types of regressions which are used in data science and machine learning. Each type has its own importance on different scenarios, but at the core, all the regression methods analyze the effect of the independent variable on dependent variables. Here we are discussing some important types of regression which are given below:
- Linear Regression
- Logistic Regression
- Polynomial Regression
- Support Vector Regression
- Decision Tree Regression
- Random Forest Regression
- Ridge Regression
- Lasso Regression