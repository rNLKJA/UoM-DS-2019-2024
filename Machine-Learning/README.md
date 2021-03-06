<img src="https://images.unsplash.com/photo-1555255707-c07966088b7b?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1332&q=80" width=100% />

[[Wikipedia](https://en.wikipedia.org/wiki/Machine_learning)] Machine learning (ML) is a field of inquiry devoted to understanding and building methods that 'learn', that is, methods that leverage data to improve performance on some set of tasks. It is seen as a part of artificial intelligence. Machine learning algorithms build a model based on sample data, known as training data, to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.

**Key points show the importance of Machine Learning**
- Rapid increment in the production of data
- Solving complex problems, which are difficult for human
- Decision-making in various sectors including finance
- Finding hidden patterns and extracting useful information from data

## Classification of Machine Learning
At a broad level, machine learning can be classified into three types:
- Supervised machine learning
- Unsupervised machine learning
- Reinforcement learning

## Machine Learning Life Cycle

Machine learning has given computer systems the ability to automatically learn without being explicitly programmed. But how does a machine learning system work? So, it can be described using the life cycle of machine learning. The machine learning life cycle is a cyclic process to build an efficient machine learning project. The main purpose of the life cycle is to find a solution to the problem or project.

The machine learning life cycle involves seven major steps, which are given below:

Gathering Data -> Data Preparation -> Data Wrangling -> Analyse Data -> Train the model -> Test the model -> Deployment

The most important thing in the complete process is to understand the problem and to know the purpose of the problem. Therefore, before starting the life cycle, we need to understand the problem because a good result depends on a better understanding of the problem.

In the complete life cycle process, to solve a problem, we can create a machine learning system called a "model", and this model is created by providing "training". But to train a model, we need data, hence cycle starts by collecting data.

- Gathering Data

    Data gathering is the first step of the machine learning life cycle. The goal of this step is to identify and obtain all data-related problems.
    
    In this step, we need to identify the different data sources, as data can be collected from various sources such as files, databases, the internet, or mobile devices. It is one of the most important steps of the life cycle. The quantity and quality of the collected data will determine the efficiency of the output. The more will be the data, the more accurate will be a prediction.
    
    This step includes the below task:
    - Identify various data sources
    - Collect data
    - Integrate the data obtained from different sources
    
    By performing the above task, we get a coherent set of data, also called a dataset. It will be used in further steps.

- Data Preparation

    After collecting the data, we need to prepare it for further steps. Data preparation is a step where we put our data into a suitable place and prepare it to use in our machine learning training. In these steps, first, we put all data together and then randomize the ordering of data.

    These steps can be further divided into two processes:
    - Data exploration
    
        It is used to understand the nature of the data that we have to work with. We need to understand the characteristics, format, and quality of data. A better understanding of data leads to an effective outcome. In this, we find Correlations, general trends, and outliers.
        
    - Data pre-processing
    
        Now the next step is preprocessing the data for its analysis.
    
- Data Wrangling

    Data wrangling is the process of cleaning and converting raw data into a useable format. It is the process of cleaning the data, selecting the variable to use, and transforming the data into a proper format to make it more suitable for analysis in the next step. It is one of the most important steps of the complete process. Cleaning of data is required to address the quality issues.

    Data we have collected don't need to be always of our use as some of the data may not be useful. In real-world applications, collected data may have various issues, including:
    - Missing values
    - Duplicate data
    - Invalid data
    - Noise

    So, we use various filtering techniques to clean the data. It is mandatory to detect and remove the above issue because it can negatively affect the quality of the outcome.

- Data Analysis

    Now the cleaned and prepared data is passed on to the analysis step. This step involves:
    - Selection of analytical techniques
    - Building models
    - Review the result
    
    This step aims to build a machine learning model to analyze the data using various analytical techniques and review the outcome. It starts with the determination of the type of the problems, where we select the machine learning techniques such as Classification, Regression, Cluster analysis, Association, etc. then build the model using prepared data, and evaluate the model.
    
    Hence, in this step, we take the data and use machine learning algorithms to build the model.

- Train the model
    
    Now the next step is to train the model, in this step we train our model to improve its performance for a better outcome of the problem.

    We use datasets to train the model using various machine learning algorithms. Training a model is required so that it can understand the various patterns, rules, and, features.
    
- Test the model

    Once our machine learning model has been trained on a given dataset, then we test the model. In this step, we check for the accuracy of our model by providing a test dataset to it.

    Testing the model determines the percentage accuracy of the model as per the requirement of the project or problem.

- Deployment

    The last step of the machine learning life cycle is deployment, where we deploy the model in the real-world system.

    If the above-prepared model is producing an accurate result as per our requirement with acceptable speed, then we deploy the model in the real system. But before deploying the project, we will check whether it is improving its performance using available data or not. The deployment phase is similar to making the final report for a project.
    
---

[**Difference between Aritifical Intelligence (AI) and Machine Learning**](./doc/diff-AI-ML.md)

---

## Data Preprocessing in Machine Learning

Data preprocessing is a process of preparing the raw data and making it suitable for a machine learning model. It is the first and crucial step while creating a machine learning model.

When creating a machine learning project, it is not always a case that we come across clean and formatted data. And while doing any operation with data, it is mandatory to clean it and put it in a formatted way. So for this, we use the data preprocessing task.

**Why do we need Data Preprocessing**

Real-world data generally contains noises, and missing values, and may be in an unusable format that cannot be directly used for machine learning models. Data preprocessing is an isan q ed tas or cleaning the data and making it suitable for a machine learning model which also increases the accuracy and efficiency of a machine learning model. It involves the below steps:
- Getting the dataset

    To create a machine learning model, the first thing we required is a dataset as a machine learning model completely works on data. The collected data for a particular problem in a proper format is known as the dataset.
    
    Datasets may be of different formats for different purposes, such as, if we want to create a machine learning model for business purposes, then the dataset will be different from the dataset required for a liver patient. So each dataset is different from another dataset. To use the dataset in our code, we usually put it into a CSV file. However, sometimes, we may also need to use HTML or xlsx. file.
    
- Importing libraries

    To perform data preprocessing using Python, we need to import some predefined Python libraries. The most common libraries: are NumPy, pandas and matplotlib.
    
- Importing datasets

    We need to import the datasets which we have collected for our machine learning project. But before importing a dataset, we need to set the current directory as a working directory. To set a working directory in Spyder IDE, we need to follow the below steps:
    - Save your Python file in the directory which contains the dataset
    - Go to the File Explorer option in IDE (Jupiter/Spyder) and select the required directory.

- Finding missing data

    The next step of data preprocessing is to handle missing data in the datasets. If our dataset contains some missing data, then it may create a huge problem for our machine learning model. Hence it is necessary to handle missing values present in the dataset.

    Ways to handle missing data: There are mainly two ways to handle missing data, which are:
    - By deleting the particular row: The first way is used to commonly deal with null values. In this way, we just delete the specific row or column which consists of null values. But this way is not so efficient and removing data may lead to loss of information which will not give an accurate output.
    - By calculating the mean: In this way, we will calculate the mean of that column or row which contains any missing value and will put it in the place of the missing value. This strategy is useful for the features which have numeric data such as age, salary, year, etc. Here, we will use this approach.
    - To handle missing values, we will use the Scikit-learn library in our code, which contains various libraries for building machine learning models.

- Encoding categorical data

    Categorical data is data that has some categories such as, in our dataset; there are two categorical variables, Country, and Purchased.

    Since the machine learning model completely works on mathematics and numbers, if our dataset would have a categorical variable, then it may create trouble while building the model. So it is necessary to encode these categorical variables into numbers.

- Splitting dataset into training and test set

    In machine learning data preprocessing, we divide our dataset into a training set and a test set. This is one of the crucial steps of data preprocessing as by doing this, we can enhance the performance of our machine learning model.

    Suppose, if we have given training to our machine learning model by a dataset and we test it with a completely different dataset. Then, it will create difficulties for our model to understand the correlations between the models.

    If we train our model very well and its training accuracy is also very high, but we provide a new dataset to it, then it will decrease the performance. So we always try to make a machine learning model which performs well with the training set and also with the test dataset. Here, we can define these datasets as:
    - Training Set: A subset of dataset to train the machine learning model, and we already know the output.
    - Test set: A subset of dataset to test the machine learning model, and by using the test set, the model predicts the output.
    
- Feature scaling

    Feature scaling is the final step of data preprocessing in machine learning. It is a technique to standardize the independent variables of the dataset in a specific range. In feature scaling, we put our variables in the same range and the same scale so that no variable dominates the other variable.

---

## Dimensionality Reduction Methods

[[Wikipedia](https://en.wikipedia.org/wiki/Dimensionality_reduction)] Dimensionality reduction, or dimension reduction, is the transformation of data from a high-dimensional space into a low-dimensional space so that the low-dimensional representation retains some meaningful properties of the original data, ideally close to its intrinsic dimension. Working in high-dimensional spaces can be undesirable for many reasons; raw data are often sparse as a consequence of the curse of dimensionality, and analyzing the data is usually computationally intractable (hard to control or deal with). Dimensionality reduction is common in fields that deal with large numbers of observations and/or large numbers of variables, such as signal processing, speech recognition, neuroinformatics, and bioinformatics.

---

### Feature selection 

[[Wikipedia](https://en.wikipedia.org/wiki/Dimensionality_reduction)] Feature selection approaches try to find a subset of the input variables (also called features or attributes). The three strategies are the filter strategy (e.g. information gain), the wrapper strategy (e.g. search guided by accuracy), and the embedded strategy (selected features are added or removed while building the model based on prediction errors).

Data analysis such as regression or classification can be done in the reduced space more accurately than in the original space.

---

### Feature projection 

[[Wikipedia](https://en.wikipedia.org/wiki/Dimensionality_reduction)] Feature projection (also called feature extraction) transforms the data from a high-dimensional space to a space of fewer dimensions. The data transformation may be linear, as in principal component analysis (PCA), but many nonlinear dimensionality reduction techniques also exist.[4][5] For multidimensional data, tensor representation can be used in dimensionality reduction through multilinear subspace learning.

- [x] [Principal Component Analysis (PCA)](./notebooks/PCA.ipynb)
- [ ] [t-distributed stochastic neighbor embedding (t-SNE)]()
- [ ] [Kernel PCA]()
- [ ] [Graph-based kernel PCA]()
- [ ] [Linear Discriminant Analysis (LDA)]()
- [ ] [Generalized Discriminant Analysis (GDA)]()
- [ ] [Autoencoder]()
- [ ] [Missing Values Ratio]()
- [ ] [Low Variance Filter]()
- [ ] [High Correlation Filter]()
- [ ] [Non-negative matrix factorization (NMF)]()
- [ ] [Uniform Manifold Approximation and Projection (UMAP)]()

---

### Dimension reduction 

[[Wikipedia](https://en.wikipedia.org/wiki/Dimensionality_reduction)] For high-dimensional datasets (i.e. with several dimensions more than 10), dimension reduction is usually performed before applying a K-nearest neighbours algorithm (k-NN) to avoid the effects of the curse of dimensionality.[20]

Feature extraction and dimension reduction can be combined in one step using principal component analysis (PCA), linear discriminant analysis (LDA), canonical correlation analysis (CCA), or non-negative matrix factorization (NMF) techniques as a pre-processing step followed by clustering by K-NN on feature vectors in reduced-dimension space. In machine learning, this process is also called low-dimensional embedding.[21]

For very-high-dimensional datasets (e.g. when performing similarity search on live video streams, DNA data or high-dimensional time series) running a fast approximate K-NN search using locality-sensitive hashing, random projection, "sketches", or other high-dimensional similarity search techniques from the VLDB conference toolbox might be the only feasible option.

---

### Evaluation Metrics 

- [x] TP, FP, TN, FN

    Performance measurement TP, TN, FP, and FN are the parameters used in the evaluation of specificity, sensitivity and accuracy.
    - True Positive or TP is the number of perfectly-identified DR pictures. 
    - True Negatives or TN is the number of perfectly detected non-DR pictures. 
    - False Positive or FP is the number of wrongly detected DR images as positive which is non-DR. 
    - False Negative or FN is the number of wrongly detected non DR which is DR. 
    
    The figure below shows the measurements using these parameters. 
    - Sensitivity is the percentage of positive cases and specificity is the percentage of negative cases. 
    - Accuracy is the percentage of correctly identified cases.
    
    By using TP, FP, TN, and FN, we can calculate the sensitivity, specificity, accuracy, precision, and negative predictive value to evaluate our machine learning model performance.
    - Sensitivity = TP / (TP + FN)
    - Specificity = TN / (FP + TN)
    - Accuracy = (TP + TN) / (TP + FN + FP + TN)
    - Precision = TP / (TP + FP)
    - Negative Predictive Value: TN / (TN + FN)
    
- [x] Confusion Matrix
    
    A confusion matrix can be used in error analysis which answers the question: of why a given model has misclassified an instance in the way it has. Using the Confusion matrix, we could:
    - Identifying different "classes" or errors that the system makes (predicted vs. actual labels).
    - Hypothesising as to what has caused the different errors, and testing those hypotheses against the actual data.
    - Quantifying whether (for different classes) it is a question of data quantity/sparsity, or something more fundamental than that.
    - Feeding those hypotheses back into feature/model engineering to see if the model can be improved.

> **Error Analysis**: Why a given model has misclassified an instance in the way it has.

> **Model Interpretability**: Why a given model has classified an instance in the way it has.

- [x] Accuracy, Precision, Recall, Specificity, F1-score

    **Accuracy**: the base metric used for model evaluation, describing the number of correct predictions overall predictions.
    - Accuracy = (TP + TN) / (TP + FN + FP + TN) = Number of correct predictions / Number of all predictions = Number of correct prediction / Size of Dataset
    
    **Precision**: a measure of how many of the positive predictions made are correct (TP). 
    - Precision = TP / (TP + FP) = Number of correctly predicted positive instances / Number of positive predictions
    
    **Recall**: a measure of how many of the positive cases the classifier correctly predicted, over all the positive cases in the data. It is also referred to as *Sensitivity*.
    - Recall = TP / (TP + FN) = Number of correctly predicted positive instances / Number of total positive instance in the dataset
    
    **Specificity**: Specificity is a measure of how many negative predictions made are correctly (true negatives).
    - Specificity = TN / (FP + TN)  = Number of correctly predicted negative instances / Number of total negative instances
    
    **F1-Score**: a measure combining both precision and recall. It is generally described as the harmonic mean of the two. Harmonic mean is just another way to calculate the "average" values, generally described as more suitable ratios (such as precision and recall) than the traditional arithmetic mean. 
    - F1-score = 2 * (Precision * Recall) / (Precision + Recall)
    - The idea is to provide a single metric that weights the two ratios (precision and recall) in a balanced way, requiring both to have a higher value for the F1-score value to rise.
    - Very small precision or recall will result in a lower overall score. Thus it helps balance the two metrics.
    - If you choose your positive class as the one with fewer samples, F1-score can help balance the metric across positive/negative samples.
    
- [ ] [Area Under the Curve - Receiver Operating Characteristics (AUC-ROC)]()
- [ ] [Log loss]()
- [ ] [Entropy]()
- [ ] [Mutual Information]()
- [ ] [Information Gain]()
- [ ] [Joint Mutual Information]()
- [ ] [Bootstrap Evaluation]()

---

### Wrapper Methods 

- [ ] [Step-wise Forward Feature Selection]()
- [ ] [Backward Feature Elimination]()
- [ ] [Exhaustive Feature Selection]()
- [ ] [Recursive Feature Elimination]()
- [ ] [Boruta]()

---

### Embedded Methods 

- [ ] [Lasso Regularization (L1)](./notebooks/supervised/regression/lasso.ipynb)
- [ ] [Ridge Regularization (L2)](./notebooks/supervised/regression/ridge.ipynb)
- [ ] [Random Forest Importance]()

---

### Ensemble Learning 

- [ ] [Bagging (Bootstrap Aggregating)]()
- [ ] [Boosting]()

---

## Data Processing Concepts 

- [ ] [One Hot Encoding]()
- [ ] [Dummy Encoding]()
- [ ] [Normalisation]()
- [ ] [Standardisation]()
- [ ] [Discretisation]()

---

## Supervised Learning Methods  
Supervised learning is a type of machine learning method in which we provide sample labelled data to the machine system to train it, and on that basis, it predicts the output.

The system creates a model using labelled data to understand the datasets and learns about each data, once the training and processing are done then we test the model by providing sample data to check whether it is predicting the exact output or not.

> Goal: Learn mappting from attributes to concepts: concept = f(attributes)

To goal of supervised machine learning is to map input data with the output data. Supervised learning is based on supervision, and it is the same as when a student learns things under the supervision of the teacher. An example of supervised learning is spam filtering.

Supervised learning can be grouped further in two categories of algorithms:

- Classification
    - [ ] [Decision Tree](./notebooks/supervised/classification/)
    - [ ] [Random Forest](./notebooks/supervised/classification/)
    - [ ] [Logistic Regression](./notebooks/supervised/classification/)
    - [ ] [K Nearest Neighbors](./notebooks/supervised/classification/KNN.ipynb)
    - [ ] [Perceptron](./notebooks/supervised/classification/Perceptron.ipynb)
    - [ ] [Navie Bayes](./notebooks/supervised/classification/NaiveBayes.ipynb)
    - [ ] [Support Vector Machine (SVM)](./notebooks/supervised/classification/SVM.ipynb)
    - [ ] [AdaBoost](./notebooks/supervised/classification/)
    - [ ] [XGBoost](./notebooks/supervised/classification/)
    - [ ] [Light GBM](./notebooks/supervised/classification/)
    - [ ] [Recommender System](https://thingsolver.com/introduction-to-recommender-systems/)

- Regression

    Regression algorithms are used if there is a relationship between the input variable and the output variable. It is used for the prediction of continuous variables, such as Weather forecasting, Market Trends, etc. Below are some popular Regression algorithms which come under supervised learning.
    
    - [x] [Regression Analysis](./doc/regression.md)
    - [ ] [Linear Regression](./notebooks/supervised/regression/LinearRegression.ipynb)
    
        - Linear regression is a statistical regression method that is used for predictive analysis.
        - It is one of the very simple and easy algorithms which works on regression and shows the relationship between the continuous variables.
        - It is used for solving the regression problem in machine learning.
        - Linear regression shows the linear relationship between the independent variable (X-axis) and the dependent variable (Y-axis), hence called linear regression.
        - If there is only one input variable (x), then such linear regression is called simple linear regression. And if there is more than one input variable, then such linear regression is called multiple linear regression.
        - The relationship between variables in the linear regression model can be explained via the mathematical equation Y = aX + b.
    
    - [ ] [Multiple Linear Regression](./notebooks/supervised/regression/)  
    - [ ] [Logistic Regression](./notebooks/supervised/regression/LogisticRegression.ipynb)
    
        - Logistic regression is another supervised learning algorithm that is used to solve classification problems. In classification problems, we have dependent variables in a binary or discrete format such as 0 or 1.
        - Logistic regression algorithm works with the categorical variable such as 0 or 1, Yes and No, True or False, Spam or not Spam, etc.
        - It is a predictive analysis algorithm that works on the concept of probability.
        - Logistic regression is a type of regression, but it is different from the linear regression algorithmtermsterm of how they are used.
        - Logistic regression uses a sigmoid function or logistic function which is a complex cost function. This sigmoid function is used to model the data in logistic regression. The function can be represented as:
            - f(x) = 1 / (1 + e^{-x})
                - f(x) = Output between the 0 and 1 value
                - x = input to the function
                - e = base of the natural logarithm
        - It uses the concept of threshold levels, values above the threshold level are rounded up to 1, and values below the threshold level are rounded up to 0.
        - There are three types of logistic regression:
            - Binary(0/1, pass/fail)
            - Multi(cats, dogs, lions)
            - Ordinal(low, medium, high)
        
    - [ ] [Backward Elimination](./notebooks/supervised/regression/)
    - [ ] [Polynomial Regression](./notebooks/supervised/regression/)
    
        - Polynomial Regression is a type of regression that models the non-linear dataset using a linear model.
        - It is similar to multiple linear regression, but it fits a non-linear curve between the value of x and corresponding conditional values of y.
        - Suppose there is a dataset that consists of data points that are present in a non-linear fashion, so for such a case, linear regression will not best fit those data points. To cover such data points, we need Polynomial regression.
        - In polynomial regression, the original features are transformed into polynomial features of a given degree and then modelled using a linear model. This means the data points are best fitted using a polynomial line.
        
        - The prediction for polynomial regression also derived from linear regression equation that means linear regression equation Y = b_0 + b_1 x is transformed into Polynomial regression equation Y = b_0 + b_1 x^2 + b_3 x^3 + ... + b_n x^n.
        - Here Y is the predicted/target output, b_0, b_1, ..., and b_n are the regression coefficients. x is our independent/input variable.
        - The model is still linear as the coefficients are still linear with quadratic.
        
        > This is different from Multiple Linear regression in such a way that in Polynomial regression, a single element has different degrees instead of multiple variables with the same degree.
    
    - [ ] [Bayesian Linear Regression](./notebooks/supervised/regression/)
    - [ ] [Support Vector Regression](./notebooks/supervised/regression/)
    - [ ] [Decision Tree Regression](./notebooks/supervised/regression/)
    
        - Decision Tree is a supervised learning algorithm that can be used for solving both classification and regression problems.
        - It can solve problems for both categorical and numerical data.
        - Decision Tree regression builds a tree-like structure in which each internal node represents the "test" for an attribute, each branch represents the result of the test, and each leaf node represents the final decision or result.
        - A decision tree is constructed starting from the root node/parent node (dataset), which splits into left and right child nodes (subsets of the dataset). These child nodes are further divided into their children nodes, and themselves become the parent node of those nodes.
        
    - [ ] [Random Forest Regression](./notebooks/supervised/regression/)
    
        - Random forest uses the Bagging or Bootstrap Aggregation technique of ensemble learning in which aggregated decision tree runs in parallel and do not interact with each other. 
        - With the help of Random Forest regression, we can prevent overfitting in the model by creating random subsets of the dataset.
        
    - [ ] [Ridge Regression](./notebooks/supervised/regression/ridge.ipynb)
    
        - Ridge regression is one of the most robust versions of linear regression in which a small amount of bias is introduced so that we can get better long-term predictions.
        - The amount of bias added to the model is known as the Ridge Regression penalty. We can compute this penalty term by multiplying the lambda by the squared weight of each feature.
        - The equation for ridge regression will be: L(x, y) = min( sum( y_i - w_ix_i)^2 + lambda sum(w_i)^2 )
        - A general linear or polynomial regression will fail if there is high collinearity between the independent variables, so to solve such problems, Ridge regression can be used.
        - Ridge regression is a regularization technique, which is used to reduce the complexity of the model. It is also called L2 regularization.
        - It helps to solve the problems if we have more parameters than samples.
        
    - [ ] [Lasso Regression](./notebooks/supervised/regression/lasso.ipynb)
        
        - Lasso regression is another regularization technique to reduce the complexity of the model.
        - It is similar to the Ridge Regression except the penalty term contains only the absolute weights instead of a square of weights.
        - Since it takes absolute values, hence, it can shrink the slope to 0, whereas the Ridge regression can only shrink it near to 0.
        - It is also called L1 regularization. The equation for Lasso regression will be: L(x, y) = min( sum(y_i - w_ix_i)^2 + lambda sum(|w_i|))
---

| Advantages of Supervised Learning                                                                               | Disadvantages of Supervised Learning                                                                          |
| --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| With the help of supervised learning, the model can predict the output based on prior experiences        | Supervised learning models are not suitable for handling the complex tasks                                   |
| In supervised learning, we can have an exact idea about the classes of objects                                  | Supervised learning cannot predict the correct output if the test data is different from the training dataset |
| Supervised learning model helps us to solve various real-world problems such as fraud detection, and spam filtering | Training requires lots of computation times                                                                  |
|                                                                                                                 | In supervised learning, we need enough knowledge about the classes of object                                  |

---

## Unsupervised Learning Methods 

Unsupervised learning is a learning method in which a machine learns without any supervision. The training is provided to the machine with the set of data that has not been labelled, classified, or categorized, and the algorithm needs to act on that data without any supervision. The goal of unsupervised learning is to restructure the input data into new features or a group of objects with similar patterns. For example:
- Unsupervised learning helps find useful insights from data
- Unsupervised learning is much similar to human learning to think by their own experiences, which makes it closer to the real AI
- Unsupervised learning works on unlabeled and uncategorized data which makes unsupervised learning more important
- In real-world, we do not always have input data with the corresponding output so to solve such cases, we need unsupervised learning 

> Goal: learn mapping from attributes to concepts: concept = f(attributes)

In unsupervised learning, we don't have a predetermined result. The machine tries to find useful insights from the huge amount of data. It can be further classified into two categories:
- Clustering
- Association
    - Detect useful patterns, associations, correlations or causal relations between attributes or between attributes and concepts.
    - A good pattern is a combination of attribute values where the presence of certain values strongly predicts the presence of other values.
    - Any kind of structure is considered interesting and there may be no "right" answer.
    - Evaluation can be difficult, potentially many possible association rules in one dataset.

- [ ] [k-means clustering (KMean)](./notebooks/unsupervised/clustering/KMean.ipynb)
- [ ] [Hierarchical Clustering](./notebooks/unsupervised/clustering/)
    - [ ] Sinlge Linkage
    - [ ] Complete Linkage
    - [ ] Average Linkage
    - [ ] Centroid Linakge
- [ ] [Anomaly detection]()
- [ ] [VAT: Visual Assessment of (Cluster) Tendency](./notebooks/unsupervised/clustering/)
- [ ] [Indenpendent Component Analysis (IDA)]()
- [ ] [Apriori algorithm]()
- [ ] [Singular value decomposition]()
- [ ] [DBSCAN](./notebooks/unsupervised/clustering/)
- [ ] [Mean Shift]()
- [ ] [OPTICS]()
- [ ] [Spectral Clustering]()
- [ ] [Mixture of Gaussians]()
- [ ] [BIRCH]()
- [ ] [Agglomerative Clustering]()
- [ ] [Neural Networks]()
- [ ] [Apriori Algorithm]()
- [ ] [Singular value decomposition]()

| Advantages of Unsupervised Learning | Disadvantages of Unsupervised Learning |
| ----------------------------------- | -------------------------------------- |
| Unsupervised learning is used for more complex tasks as compared to supervised learning because, in unsupervised learning, we don't have labelled input data | Unsupervised learning is intrinsically more difficult than supervised learning as it does not have corresponding output |
| Unsupervised learning is preferable as it is easy to get unlabeled data in comparison to labelled data | The result of the unsupervised learning algorithm might be less accurate as input data is not labelled, and algorithms do not know the exact output in advance |

## Supervised learning vs. Unsupervised learning
    
|            | Supervised Learning              | Unsupervised Learning    |
| ---------- | -------------------------------- | ------------------------ |
| Discrete   | Classification<br>Categorization | Clustering               |
| Continuous | Regression                       | Dimensionality Reduction |

| Supervised learning | Unsupervised learning |
| ------------------- | --------------------- |
| Supervised learning algorithms are trained using labelled data | Unsupervised learning algorithms are trained using unlabeled data |
| Supervised learning model takes direct feedback to check if it is predicting correct output or not | Unsupervised learning model does not take any feedback |
| Supervised learning model predicts the output | Unsupervised learning model finds the hidden patterns in data |
| In supervised learning, input data is provided to the model along with the output | In unsupervised learning, only input data is provided to the model |
| The goal of supervised learning is to train the model so that it can predict the output when it is given new data | The goal of unsupervised learning is to find the hidden patterns and useful insights from the unknown dataset |
| Supervised learning needs supervision to train the model | Unsupervised learning does not need any supervision to train the model |
| Supervised learning can be categorized in Classification and Regression problems | Unsupervised Learning can be classified in Clustering and Associations problems |
| Supervised learning can be used for those cases where we know the input as well as corresponding outputs | Unsupervised learning can be used for those cases where we have only input data and no corresponding output data |
| Supervised learning model produces an accurate result | Unsupervised learning model may give less accurate result as compared to supervised learning |
| Supervised learning is not close to true Artificial intelligence as in this, we first train the model for each data, and then only it can predict the correct output | Unsupervised learning is more close to the true Artificial Intelligence as it learns similarly to a child learns daily routine things by his experiences |
| It includes various algorithms such as Linear Regression, Logistic Regression, Support Vector Machine, Multi-class Classification, Decision tree, Bayesian Logic, etc. | It includes various algorithms such as Clustering, KNN, and Apriori algorithm |

---

## Marr's Levels of Analysis
Framework for understanding information processing systems.
- Computational Level: what is the goal of this system.
    - What structure does this machine learning model expect to see in the world?
    - What rule/pattern/model/etc. explains this data?
- Algorithm Level: How do you achieve the goal, algorithms and data structure?
    - Given a model, what's the best fit for this data?
    - Usually involves minimizing an error or loss function.
- Implementation Level: Physical implementation (circuits, neurons).
    - How to find that best fit in finite time.
    It- Not always possible to solve exactly.

---

> **About model assumptions**: What kinds of assumptions might a machine learning model make then tackle these problems?

Every model makes assumptions about the world and how the concepts we want to learn to relate to the attributes of the data.
- *The first assumption we make is that the concept is related to the attributes?*

    This assumption is so obvious that we rarely discuss it ??? usually, we only include attributes that we think are likely to predict the concept. For example, you would probably not use ???patient???s favourite song??? as an attribute for skin cancer detection. However, this attribute might be a good predictor, because your favourite song can be a good predictor of your age, and age is a risk factor for skin cancer. You could probably come up with other ???weird??? predictors for each of the example models.

- *Secondly, each model makes assumptions about the ways the attributes can relate to the concepts.*

    For example, does it make more sense for the models to treat all attributes as independent predictors, or would it be better to use a model that allows the predictors to interact? In most of these cases, we would expect the attributes to interact in complex ways but allowing interactions could lead to an overly complex model in the cases where there are many attributes to start with (for example, in the customer purchasing model). For the problems with numeric attributes, would we generally expect linear (or monotonic, e.g., strictly increasing or decreasing) relationships between the attributes and concepts? This is often a good simplifying assumption for machine learning, but it limits what a model can learn. For example, the relationship between ???best burrito??? and price might be U-shaped ??? very cheap and very expensive burritos might be less popular than burritos priced somewhere in the middle.
