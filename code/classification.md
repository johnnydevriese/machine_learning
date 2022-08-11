# Logistic Regression  

References: 
https://www.kdnuggets.com/2020/01/guide-precision-recall-confusion-matrix.html
https://developers.google.com/machine-learning/crash-course/classification/thresholding
https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/ 

https://developers.google.com/machine-learning/crash-course/logistic-regression/model-training

### Thresholding 

A logistic regression model that returns 0.9995 for a particular email message is predicting that it is very likely to be spam. Conversely, another email message with a prediction score of 0.0003 on that same logistic regression model is very likely not spam. However, what about an email message with a prediction score of 0.6? In order to map a logistic regression value to a binary category, you must define a classification threshold (also called the decision threshold). A value above that threshold indicates "spam"; a value below indicates "not spam." It is tempting to assume that the classification threshold should always be 0.5, but thresholds are problem-dependent, and are therefore values that you must tune.

### Accuracy 

$ 
Accuracy = Total correct / total predictions
$ 

Using the Confusion Matrix values

$ 
Accuracy = TP + TN / TP + FP + TN + FN
$

Accuracy alone doesn't tell the full story when you're working with a **class-imbalanced data** set, like this one, where there is a significant disparity between the number of positive and negative labels.

### Precision 

Precision — Also called Positive predictive value
The ratio of correct positive predictions to the *total predicted positives.*

$
Precision = \frac{TP}{TP + FP}
$


### Recall 

Recall — Also called Sensitivity, Probability of Detection, True Positive Rate

The ratio of correct positive predictions to the *total positives examples.*

$
Recall = \frac{TP}{TP + FN}
$

### ROC & AUC 

* ROC Curves summarize the trade-off between the true positive rate and false positive rate for a predictive model using different probability thresholds.
    
* Precision-Recall curves summarize the trade-off between the true positive rate and the positive predictive value for a predictive model using different probability thresholds.

* ROC curves are appropriate when the observations are balanced between each class, whereas precision-recall curves are appropriate for imbalanced datasets.


### sklearn functions

https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html


### What about using XGBoost for classification? 

We can still use XGBoost but logistic regression is linear and XGBoost is *not* linear. 

For example we can see here that we are drawing linear boundaries between classifications in the iris dataset. 

https://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html#sphx-glr-auto-examples-linear-model-plot-iris-logistic-py 


### What about difference between SVM and logistic regression? 


### Logistic Regression Loss function 

**This always trips me up because some people call it *log loss* or cross entropy or logits or something else!**

The loss function for linear regression is squared loss. The loss function for logistic regression is Log Loss, which is defined as follows:
$
put formula in here later
$

is the data set containing many labeled examples, which are
pairs.
is the label in a labeled example. Since this is logistic regression, every value of
must either be 0 or 1.
is the predicted value (somewhere between 0 and 1), given the set of features in . 
