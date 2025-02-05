# 7.1 Basics 

1. [E] Explain supervised, unsupervised, weakly supervised, semi-supervised, and active learning.

**Supervised**: "Applications in which the training data comprises examples of the input vectors along with their corresponding target vectors are known as supervised learning problems." (Bishop)

example: regression 

**Unsupervised**: In unsupervised learning, there is no instructor or teacher, and the algorithm must learn to make sense of the data without this guide.

example: clustering (k-means)

**Weakly Supervised**: Weak supervision is a branch of machine learning where noisy, limited, or imprecise sources are used to provide supervision signal for labeling large amounts of training data in a supervised learning setting.[1] This approach alleviates the burden of obtaining hand-labeled data sets, which can be costly or impractical. Instead, inexpensive weak labels are employed with the understanding that they are imperfect, but can nonetheless be used to create a strong predictive model.

**Self-Supervised**: 

Self-supervised learning refers to an unsupervised learning problem that is framed as a supervised learning problem in order to apply supervised learning algorithms to solve it.

Supervised learning algorithms are used to solve an alternate or pretext task, the result of which is a model or representation that can be used in the solution of the original (actual) modeling problem.

A general example of self-supervised learning algorithms are autoencoders. These are a type of neural network that is used to create a compact or compressed representation of an input sample. They achieve this via a model that has an encoder and a decoder element separated by a bottleneck that represents the internal compact representation of the input.

2. Empirical risk minimization.
[E] What’s the risk in empirical risk minimization?
[E] Why is it empirical?
[E] How do we minimize that risk?

Maximum Likelihood Estimation (MLE) is a special case of ERM. The loss of the hypothesis function 

$$

R_{emp} = \frac{1}{n}\sum_{i=1}^{n} Loss(h(x_i), y_i)

$$

where $h$ is our hypothesis function 


The ERM principal states learning also should choose a hypothesis function $\hat{h}$ that minimizes the empirical risk. 

$$
\hat{h} = \arg\min_{h \in \mathcal{H}}  R_{\text{emp}} h
$$



3. Occam's razor states that when the simple explanation and complex explanation both work equally well, the simple explanation is usually correct. How do we apply this principle in ML?

With that in mind, some experts feel that Occam's razor can be useful and instructive in designing machine learning projects. Some contend that Occam's razor can help engineers to choose the best algorithm to apply to a project, and also help with deciding how to train a program with the selected algorithm. One interpretation of Occam's razor is that, given more than one suitable algorithm with comparable trade-offs, the one that is least complex to deploy and easiest to interpret should be used.

Others point out that simplification procedures such as feature selection and dimensionality reduction are also examples of using an Occam's razor principle – of simplifying models to get better results. On the other hand, others describe model trade-offs where engineers reduce complexity at the expense of accuracy – but still argue that this Occam's razor approach can be beneficial.


[source](https://www.techopedia.com/how-does-occams-razor-apply-to-machine-learning/7/33087)

4.[E] What are the conditions that allowed deep learning to gain popularity in the last decade?

larger datasets available and also rise of GPU compute. 

5. [M] If we have a wide NN and a deep NN with the same number of parameters, which one is more expressive and why?

A deep NN will be because it has more non linearities available for modeling. 

6. [H] The Universal Approximation Theorem states that a neural network with 1 hidden layer can approximate any continuous function for inputs within a specific range. Then why can’t a simple neural network reach an arbitrarily small positive error?


Universal approximation theorems imply that neural networks can represent a wide variety of interesting functions when given appropriate weights. On the other hand, they typically do not provide a construction for the weights, but merely state that such a construction is possible.

One hidden layer can't model complex nonlinear relationships and therefore can't get a small positive error. 

7. [E] What are saddle points and local minima? Which are thought to cause more problems for training large NNs?

NNs are universal approximators but aren't guaranteed to reach the global minimum. So, using gradient descent when training you might find local minima or saddle points. Saddle points cause more problems for training.  

8. Hyperparameters.
* [E] What are the differences between parameters and hyperparameters?
* [E] Why is hyperparameter tuning important?
* [M] Explain algorithm for tuning hyperparameters.


Model Parameters: These are the parameters in the model that must be determined using the training data set. These are the fitted parameters. example: weights/biases 

Hyperparameters: These are adjustable parameters that must be tuned in order to obtain a model with optimal performance. example: learning rate, number of iterations, number of layers, number of clusters in k-clustering. 

9. Classification vs. regression.
* [E] What makes a classification problem different from a regression problem?
* [E] Can a classification problem be turned into a regression problem and vice versa?


Classification is the task of predicting a discrete class label.
Regression is the task of predicting a continuous quantity.

In some cases you could convert target label into a regression quantity and then use regression instead of classification. 

[source](https://machinelearningmastery.com/classification-versus-regression-in-machine-learning)

10. Parametric vs. non-parametric methods.
[E] What’s the difference between parametric methods and non-parametric methods? Give an example of each method.
[H] When should we use one and when should we use the other?


parameteric: set number of parameters (weights). So a neural net is parametric. 

nonparametric: a varying number of parameters. XGBoost is an example because we can grow the tree infinitely. 


[source](http://manjeetdahiya.com/posts/parametric-vs-non-parametric-models/)


11. [M] Why does ensembling independently trained models generally improve performance?


"The idea behind a random forest is to average multiple (deep) decision tress that individually suffer from high variance to build a more robust model that has a better generalization performacne is less susceptible to overfitting." - Raschka p.95 

Majority voting is most common (and probably most intuitive.)

See chapter 7 in Raschka's book for more on ensemble methods. 

Scikit learn also has a nice explanation about what ensemble methods are but not great on *why* they improve performance. 

[sckit learn ensemble](https://scikit-learn.org/stable/modules/ensemble.html)


I suppose it's probably the *wisdom of the crowd*. Since each model picks up specific features and if you average them all out it you can find a nice balance between bias/variane. 

12. [M] Why does L1 regularization tend to lead to sparsity while L2 regularization pushes weights closer to 0?

L1 is sum of weights and 
$$
\|x\|_{1}:=\sum _{i=1}^{n}\left|x_{i}\right|
$$

L2 norm (Euclidian - ordinary sum of squares) 
$$ 
\|x\|_{2} := \sqrt{x_{1}^{2}+\cdots +x_{n}^{2}}
$$


[L1 regularization](https://www.educative.io/answers/why-does-l1-regularization-yield-sparse-solutions)


We can see pictorially in _Machine Learning with PyTorch and SciKit-Learn_ (p. 123) that the L2 norm creates a circle/ball that is our regularization budget and will pull weights to zero as we make the regularization term larger. 

For an L1 norm it is a rotated(45 degree) square so that we can choose a point where we have large value for one weight (say $w_2$ in example) and zero weight ($w_1$) in another.

Raschka notes to look into ESL by Hastie Ribshirani et al for more on why l1 normalization creates sparse solutions. (IIRC Hastie created LASSO...)

Note: l1 regularization can be thought of as a technique for feature selection. 

13. [E] Why does an ML model’s performance degrade in production?

Concept drift: means that the statistical properties of the target variable, which the model is trying to predict, change over time in unforeseen ways. This causes problems because the predictions become less accurate as time passes.

[source](https://towardsdatascience.com/why-machine-learning-models-degrade-in-production-d0f2108e9214)


Models can only work on the data it has seen and so if the data changes or we get more data the performance might not work as well. Especially if the model hasn't generalized well. 

14. [M] What problems might we run into when deploying large machine learning models?

where to deploy, monitor and store everything. Organizational difficulties in getting everything set up. 
Performance of serving model. also worrying about ethics/biases in the model. 

15. Your model performs really well on the test set but poorly in production.
* [M] What are your hypotheses about the causes?
* [H] How do you validate whether your hypotheses are correct?
* [M] Imagine your hypotheses about the causes are correct. What would you do to address them?


Model has probably been over fit. You could re train model with different hyperparameters and test against the test/validation sets to compare with underfitting. If model with different parameters seems to generalize better on the validation test set then we could deploy it to production and monitor performance. 

