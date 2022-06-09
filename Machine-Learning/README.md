<img src="./img/arseny-togulev-MECKPoKJYjM-unsplash.jpg" width=100% />

<div align=center><h3>Machine Learning</h3></div>

[[Wikipedia](https://en.wikipedia.org/wiki/Machine_learning)] Machine learning (ML) is a field of inquiry devoted to understanding and building methods that 'learn', that is, methods that leverage data to improve performance on some set of tasks.[1] It is seen as a part of artificial intelligence. Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so.[2] Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.

---

<div align=center><h4>Dimensionality Reduction Methods</h4></div>

[[Wikipedia](https://en.wikipedia.org/wiki/Dimensionality_reduction)] Dimensionality reduction, or dimension reduction, is the transformation of data from a high-dimensional space into a low-dimensional space so that the low-dimensional representation retains some meaningful properties of the original data, ideally close to its intrinsic dimension. Working in high-dimensional spaces can be undesirable for many reasons; raw data are often sparse as a consequence of the curse of dimensionality, and analyzing the data is usually computationally intractable (hard to control or deal with). Dimensionality reduction is common in fields that deal with large numbers of observations and/or large numbers of variables, such as signal processing, speech recognition, neuroinformatics, and bioinformatics.

---

<div align=center><h5>Feature selection</h5></div>


[[Wikipedia](https://en.wikipedia.org/wiki/Dimensionality_reduction)] Feature selection approaches try to find a subset of the input variables (also called features or attributes). The three strategies are: the filter strategy (e.g. information gain), the wrapper strategy (e.g. search guided by accuracy), and the embedded strategy (selected features are added or removed while building the model based on prediction errors).

Data analysis such as regression or classification can be done in the reduced space more accurately than in the original space.

---

<div align=center><h5>Feature projection</h5></div>

[[Wikipedia](https://en.wikipedia.org/wiki/Dimensionality_reduction)] Feature projection (also called feature extraction) transforms the data from the high-dimensional space to a space of fewer dimensions. The data transformation may be linear, as in principal component analysis (PCA), but many nonlinear dimensionality reduction techniques also exist.[4][5] For multidimensional data, tensor representation can be used in dimensionality reduction through multilinear subspace learning.

- [ ] [Principal Component Analysis (PCA)](./notebooks/PCA.ipynb)
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

<div align=center><h5>Dimension reduction</h5></div>

[[Wikipedia](https://en.wikipedia.org/wiki/Dimensionality_reduction)] For high-dimensional datasets (i.e. with number of dimensions more than 10), dimension reduction is usually performed prior to applying a K-nearest neighbors algorithm (k-NN) in order to avoid the effects of the curse of dimensionality.[20]

Feature extraction and dimension reduction can be combined in one step using principal component analysis (PCA), linear discriminant analysis (LDA), canonical correlation analysis (CCA), or non-negative matrix factorization (NMF) techniques as a pre-processing step followed by clustering by K-NN on feature vectors in reduced-dimension space. In machine learning this process is also called low-dimensional embedding.[21]

For very-high-dimensional datasets (e.g. when performing similarity search on live video streams, DNA data or high-dimensional time series) running a fast approximate K-NN search using locality-sensitive hashing, random projection, "sketches", or other high-dimensional similarity search techniques from the VLDB conference toolbox might be the only feasible option.

---

<div align=center><h4>Evaluation Metrics</h4></div>

- [x] TP, FP, TN, FN

    Performance measurement TP, TN, FP, FN are the parameters used in the evaluation of specificity, sensitivity and accuracy.
    - True Positive or TP is the number of perfectly identified DR pictures. 
    - True Negatives or TN is the number of perfectly detected non DR picures. 
    - False Positive or FP is the number of wrongly detected DR images as positive which is actually non DR. 
    - False Negative or FN is the number of wrongly detected non DR which is actually DR. 
    
    The figure below shows the measurements using these parameters. 
    - Sensitivity is the percentage of positive cases and specificity is the percentage of negative cases. 
    - Accuracy is the percentage of correctly identified cases.

    <img src="./img/Performance-measurement-TP-TN-FP-FN-are-the-parameters-used-in-the-evaluation-of.png" align=center />
    
    By using TP, FP, TN, FN, we can calculate the sensitivity, specificity, accuracy, precision, negative predictive value to evaluate our machine learning model performance.
    - Sensitivity = TP / (TP + FN)
    - Specificity = TN / (FP + TN)
    - Accuracy = (TP + TN) / (TP + FN + FP + TN)
    - Precision = TP / (TP + FP)
    - Negative Predictive Value: TN / (TN + FN)
    
- [x] Confusion Matrix
    
    Confusion matrix can be usd in error analysis which answer the question: why a given model has misclassified an instance in the way it has. Use Confusion matrix, we could:
    - Identifying different "classes" or error that the system makes (predicted vs. actual labels).
    - Hypothesising as to what has caused the different errors, and testing those hypotheses against the actual data.
    - Quantifying whether (for different classes) it is a question of data quantity/sparsity, or something more fundamental than that.
    - Feeding those hypotheses back into feature/model engineering to see if the model can be improved.

> **Error Analysis**: Why a given model has misclassified an instance in the way it has.

> **Model Interpretability**: Why a given model has classified an instance in the way it has.

- [ ] [Precision, Recall, F1-score]()
- [ ] [Area Under the Curve - Receiver Operating Characteristics (AUC-ROC)]()
- [ ] [Log loss]()
- [ ] [Entropy]()
- [ ] [Mutual Information]()
- [ ] [Information Gain]()
- [ ] [Joint Mutual Information]()
- [ ] [Bootstrap Evaluation]()

---

<div align=center><h4>Wrapper Methods</h4></div>

- [ ] [Step-wise Forward Feature Selection]()
- [ ] [Backward Feature Elimination]()
- [ ] [Exhaustive Feature Selection]()
- [ ] [Recursive Feature Elimination]()
- [ ] [Boruta]()

---

<div align=center><h4>Embeded Methods</h4></div>

- [ ] [Lasso Regularization (L1)]()
- [ ] [Ridge Regularization (L2)]()
- [ ] [Random Forest Importance]()

---

<div align=center><h4>Ensemble Learning</h4></div>

- [ ] [Bagging (Bootstrap Aggregating)]()
- [ ] [Boosting]()

---

<div align=center><h4>Data Processing Concepts</h4></div>

- [ ] [One Hot Encoding]()
- [ ] [Dummy Encoding]()
- [ ] [Normalisation]()
- [ ] [Standardisation]()
- [ ] [Discretisation]()

---

<div align=center><h4>Supervised Learning Methods</h4></div>
 
- [ ] [K Nearest Neighbors](./notebooks/SL/KNN.ipynb)
- [ ] [Regression](./notebooks/SL/Regression.ipynb)
    - [ ] [Linear Regression](./notebooks/SL/LinearRegression.ipynb)
    - [ ] [Simple Linear Regression]()
    - [ ] [Multiple Linear Regression]()  
    - [ ] [Logistic Regression](./notebooks/SL/LogisticRegression.ipynb)
    - [ ] [Backward Elimination]()
    - [ ] [Polynomial Regression]()
- [ ] [Perceptron](./notebooks/SL/Perceptron.ipynb)
- [ ] [Navie Bayes](./notebooks/SL/NaiveBayes.ipynb)
- [ ] [Support Vector Machine (SVM)](./notebooks/SL/SVM.ipynb)
- [ ] [Decision Tree]()
- [ ] [Random Forest]()
- [ ] [AdaBoost]()
- [ ] [XGBoost]()
- [ ] [Light GBM]()
- [ ] [Recommender System](https://thingsolver.com/introduction-to-recommender-systems/)
    
    <img src="./img/types-of-recommender-systems.png" align=center />
    
---

<div align=center><h4>Unsupervised Learning Methods</h4></div>

- [ ] [k-means clustering (KMean)](./notebooks/KMean.ipynb)
- [ ] [Hierarchical Clustering]()
    - [ ] Sinlge Linkage
    - [ ] Complete Linkage
    - [ ] Average Linkage
    - [ ] Centroid Linakge
- [ ] [Anomaly detection]()
- [ ] [VAT: Visual Assessment of (Cluster) Tendency]()
- [ ] [Indenpendent Component Analysis (IDA)]()
- [ ] [Apriori algorithm]()
- [ ] [Singular value decomposition]()
- [ ] [DBSCAN]()
- [ ] [Mean Shift]()
- [ ] [OPTICS]()
- [ ] [Spectral Clustering]()
- [ ] [Mixture of Gaussians]()
- [ ] [BIRCH]()
- [ ] [Agglomerative Clustering]()

- [ ] Supervised learning vs. Unsupervised learning
    
    |            | Supervised Learning              | Unsupervised Learning    |
    | ---------- | -------------------------------- | ------------------------ |
    | Discrete   | Classification<br>Categorization | Clustering               |
    | Continuous | Regression                       | Dimensionality Reduction |
---

<img src="./img/fabio-oyXis2kALVg-unsplash.jpg" width=100%>

<div align=center><h3>Neural Network</h3></div>

- [ ] [Neuron]()
- [ ] [Layers]()
- [ ] [Epoch]()
- [ ] [Neural Network]()
- [ ] [Convolutional Neural Network]()
- [ ] [Genetic Algorithm]()

---

<div align=center><h4>Activation Functions</h4></div>

- [ ] [Linear Activation]()
- [ ] [Heaviside Activation]()
- [ ] [Logistic Activation]()
- [ ] [Sigmoid]()
- [ ] [Rectified Linear Unit (ReLU)]()
- [ ] [tanh]()
- [ ] [Softmax]()
- [ ] [Auto Encoder]()
- [ ] [Genetric Algorithm]()
- [ ] [Ensembler]()
