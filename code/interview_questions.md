https://brainstation.io/career-guides/machine-learning-engineer-interview-questions

## What’s the trade-off between bias and variance?

https://machinelearningmastery.com/gentle-introduction-to-the-bias-variance-trade-off-in-machine-learning/

low bias ML algos: decision trees, k-nearest neighbors and support vector machines 
high bias ML aglos: linear regression, linear discriminant analysis and logistic regression 



http://cs229.stanford.edu/summer2020/BiasVarianceAnalysis.pdf

Andrew Ng talks about this in CS229. (need to double check)

**bias** is about what the model is assuming about the data 

**variance** is about number of features in the data?? idk this doesnt seem right. 

## How is KNN different from k-means clustering?

k means
https://dzone.com/articles/10-interesting-use-cases-for-the-k-means-algorithm
https://www.wikiwand.com/en/K-means_clustering
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html


k-nearest neighbors
https://towardsdatascience.com/k-nearest-neighbors-knn-algorithm-23832490e3f4
https://www.wikiwand.com/en/K-nearest_neighbors_algorithm
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

## What is cross validation and what are different methods of using it? 

https://towardsdatascience.com/understanding-8-types-of-cross-validation-80c935a4976d

Leave p out cross-validation
Leave one out cross-validation
Holdout cross-validation
Repeated random subsampling validation
k-fold cross-validation
Stratified k-fold cross-validation
Time Series cross-validation
Nested cross-validation

## Explain how a ROC curve works.

Aside: it says that it is for all thresholds of classifications but it's not like that's a parameter of logistic regression model. This answers that you can just change it after the fact. 

https://towardsdatascience.com/classification-metrics-thresholds-explained-caff18ad2747

```python
# Adjusting the threshold down from 0.5 to 0.25
# Any data point with a probability  of 0.25 or higher will be 
#  classified as 1. clf = LogisticRegression()
clf.fit(X_train, y_train)
THRESHOLD = 0.25
y_pred = np.where(clf.predict_proba(X_test)[:,1] >= THRESHOLD, 1, 0)
y_pred_proba_new_threshold = (clf.predict_proba(X_test)[:,1] >= THRESHOLD).astype(int)
```

The Receiver Operator Characteristic (ROC) curve is an evaluation metric for binary classification problems. It is a probability curve that plots the TPR against FPR at various threshold values and essentially separates the ‘signal’ from the ‘noise’. The Area Under the Curve (AUC) is the measure of the ability of a classifier to distinguish between classes and is used as a summary of the ROC curve.

https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/

## What's the difference between "likelihood" and "probability"

Probability quantifies anticipation (of outcome), likelihood quantifies trust (in model).

Suppose somebody challenges us to a 'profitable gambling game'. Then, probabilities will serve us to compute things like the expected profile of your gains and loses (mean, mode, median, variance, information ratio, value at risk, gamblers ruin, and so on). In contrast, likelihood will serve us to quantify whether we trust those probabilities in the first place; or whether we 'smell a rat'.

Incidentally -- since somebody above mentioned the religions of statistics -- I believe likelihood ratio to be an integral part of the Bayesian world as well as of the frequentist one: In the Bayesian world, Bayes formula just combines prior with likelihood to produce posterior.

https://stats.stackexchange.com/questions/2641/what-is-the-difference-between-likelihood-and-probability


## How to prune decision trees? 

1. preprocessing - early stopping 
2. post processing - fit tree perfectly and then prune it back 

https://www.kaggle.com/arunmohan003/pruning-decision-trees-tutorial


## how can you choose a  classifier based on a training set size ? 

https://www.researchgate.net/post/How-to-decide-the-best-classifier-based-on-the-data-set-provided 

As far as I know there is no a well defined rule for such task. In general, it depends on the kind of data and amount of samples x features. For instance, I would recommend to use naive Bayes or linear SVM for text classification/categorization. For datasets with numerical attributes: I would suggest linear SVM, neural networks or logistic regression if the amount of features is much greater than the number of samples. On the other hand, I would recommend neural networks or SVM with RBF or polynomial kernel if the amount of samples is not too large and greater than the number of features. Otherwise, if the number of samples is huge I would suggest to use neural networks or linear SVM, and so on. Obviously, there are other options for each scenario than those I have mentioned.

## What methods for dimensionality reduction do you know and how do they compare with each other?

https://towardsdatascience.com/11-dimensionality-reduction-techniques-you-should-know-in-2021-dcb9500d388b

big ones seem to be: 
The Principal Component Analysis (PCA) procedure is a dimension reduction technique that projects the data on kkk dimensions by maximizing the variance of the data as follows: 

https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-unsupervised-learning#dimension-reduction

Independent Component Analysis (ICA) 
It is a technique meant to find the underlying generating sources.

LDA 
https://www.wikiwand.com/en/Linear_discriminant_analysis

LDA is also closely related to principal component analysis (PCA) and factor analysis in that they both look for linear combinations of variables which best explain the data.[4] LDA explicitly attempts to model the difference between the classes of data. PCA, in contrast, does not take into account any difference in class, and factor analysis builds the feature combinations based on differences rather than similarities. Discriminant analysis is also different from factor analysis in that it is not an interdependence technique: a distinction between independent variables and dependent variables (also called criterion variables) must be made. 

## What’s an imbalanced dataset? Can you list some ways to deal with it?

Any dataset with an unequal class distribution is technically imbalanced.

Here are some techniques to handle imbalanced data:

Resample the training set: There are two approaches to make a balanced dataset out of an imbalanced one are under-sampling and over-sampling.
Generate synthetic samples: Using SMOTE (Synthetic Minority Oversampling Technique) to generate new and synthetic data to train the model.

# More Questions 

https://elitedatascience.com/machine-learning-interview-questions-answers#:~:text=21%20Machine%20Learning%20Interview%20Questions%20and%20Answers%201,learning%20can%20help%20different%20types%20of%20businesses.%20


## Explain the Bias-Variance Tradeoff.

https://elitedatascience.com/bias-variance-tradeoff

Predictive models have a tradeoff between bias (how well the model fits the data) and variance (how much the model changes based on changes in the inputs).

Simpler models are stable (low variance) but they don't get close to the truth (high bias).

More complex models are more prone to being overfit (high variance) but they are expressive enough to get close to the truth (low bias).

The best model for a given problem usually lies somewhere in the middle.



