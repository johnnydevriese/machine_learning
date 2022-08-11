https://stats.stackexchange.com/questions/11859/what-is-the-difference-between-multiclass-and-multilabel-problem

https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/

https://web.stanford.edu/~nanbhas/blog/sigmoid-softmax/


Binary Classification: One node, sigmoid activation.
Multiclass Classification: One node per class, softmax activation.
Multilabel Classification: One node per class, sigmoid activation.

Multi-class vs Binary-class is the question of the number of classes your classifier is modeling. In theory, a binary classifier is much simpler than multi-class, so it's important to make this distinction. For example, the Support vector machine (SVM) can trivially learn a hyperplane to separate two classes, but 3 or more classes makes it complex. In the neural networks, we commonly use Sigmoid for binary, but Softmax for multi-class as the last layer of the model.

Multi-label vs Single-Label is the question of how many classes any object or example can belong to. In the neural networks, if single label is needed we use a single Softmax layer as the last layer, thus learning a single probability distribution that spans across all classes. If the multi-label classification is needed, we use multiple Sigmoids on the last layer, thus learning separate distribution for each class.

# Cross-Entropy or Log Likelihood in Output Layer (StackExchange)

*negative log likelihood* is also known as multi class cross-entropy 


### all of the normal loss functions and their applications in PyTorch 

https://neptune.ai/blog/pytorch-loss-functions


https://medium.com/deeplearningmadeeasy/negative-log-likelihood-6bd79b55d8b6 

It’s a cost function that is used as loss for machine learning models, telling us how bad it’s performing, the lower the better.

I’m going to explain it word by word, hopefully that will make it. easier to understand.

Negative: obviously means multiplying by -1. What? The loss of our model. Most machine learning frameworks only have minimization optimizations, but we want to maximize the probability of choosing the correct category.

We can **maximize by minimizing the negative log likelihood,** there you have it, we want somehow to maximize by minimizing.

Also it’s much easier to reason about the loss this way, to be consistent with the rule of loss functions approaching 0 as the model gets better.


cross entropy loss is same as negative log likelihood 


NLL uses a negative connotation since the probabilities (or likelihoods) vary between zero and one, and the logarithms of values in this range are negative. In the end, the loss value becomes positive.



