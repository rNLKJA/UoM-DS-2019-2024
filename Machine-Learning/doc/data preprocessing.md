### Why do we need Data Preprocessing

A real-world data generally contains noises, missing values, and maybe in an unusable format which cannot be directly used for machine learning models. Data preprocessing is required tasks for cleaning the data and making it suitable for a machine learning model which also increases the accuracy and efficiency of a machine learning model. It involves below steps:
- Getting the dataset

    To create a machine learning model, the first thing we required is a dataset as a machine learning model completely works on data. The collected data for a particular problem in a proper format is known as the dataset.
    
    Dataset may be of different formats for different purposes, such as, if we want to create a machine learning model for business purpose, then dataset will be different with the dataset required for a liver patient. So each dataset is different from another dataset. To use the dataset in our code, we usually put it into a CSV file. However, sometimes, we may also need to use an HTML or xlsx. file.
    
- Importing libraries

    In order to perform data preprocessing using Python, we need to import some predefined Python libraries. Most common libraries are: numpy, pandas and matplotlib.
    
- Importing datasets

    We need to import the datasets which we have collected for our machine learning project. But before importing a dataset, we need to set the current directory as a working directory. To set a working directory in Spyder IDE, we need to follow the below steps:
    - Save your Python file in the directory which contains dataset
    - Go to File explorer option in IDE (jupyter/Spyder) and select the required directory.

- Finding missing data

    The next step of data preprocessing is to handle missing data in the datasets. If our dataset contains some missing data, then it may create a huge problem for our machine learning model. Hence it is necessary to handle missing values present in the dataset.

    Ways to handle missing data: There are mainly two ways to handle missing data, which are:
    - By deleting the particular row: The first way is used to commonly deal with null values. In this way, we just delete the specific row or column which consists of null values. But this way is not so efficient and removing data may lead to loss of information which will not give the accurate output.
    - By calculating the mean: In this way, we will calculate the mean of that column or row which contains any missing value and will put it on the place of missing value. This strategy is useful for the features which have numeric data such as age, salary, year, etc. Here, we will use this approach.
    - To handle missing values, we will use Scikit-learn library in our code, which contains various libraries for building machine learning models.

- Encoding categorical data

    Categorical data is data which has some categories such as, in our dataset; there are two categorical variable, Country, and Purchased.

    Since machine learning model completely works on mathematics and numbers, but if our dataset would have a categorical variable, then it may create trouble while building the model. So it is necessary to encode these categorical variables into numbers.

- Splitting dataset into training and test set

    In machine learning data preprocessing, we divide our dataset into a training set and test set. This is one of the crucial steps of data preprocessing as by doing this, we can enhance the performance of our machine learning model.

    Suppose, if we have given training to our machine learning model by a dataset and we test it by a completely different dataset. Then, it will create difficulties for our model to understand the correlations between the models.

    If we train our model very well and its training accuracy is also very high, but we provide a new dataset to it, then it will decrease the performance. So we always try to make a machine learning model which performs well with the training set and also with the test dataset. Here, we can define these datasets as:

    ![data-preprocessing-machine-learning-5](./img/data-preprocessing-machine-learning-5.png)
    
    Training Set: A subset of dataset to train the machine learning model, and we already know the output.

    Test set: A subset of dataset to test the machine learning model, and by using the test set, model predicts the output.
    
- Feature scaling

    Feature scaling is the final step of data preprocessing in machine learning. It is a technique to standardize the independent variables of the dataset in a specific range. In feature scaling, we put our variables in the same range and in the same scale so that no any variable dominate the other variable.
