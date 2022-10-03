# 7.2 Sampling and creating training data

[aside: nice list of stats questions](https://towardsdatascience.com/40-statistics-interview-problems-and-answers-for-data-scientists-6971a02b7eee#:~:text=Handling%20missing%20data%20can%20make,as%20it%20might%20actually%20be.)

1. [E] If you have 6 shirts and 4 pairs of pants, how many ways are there to choose 2 shirts and 1 pair of pants?

6 * 4 = 24 total outfits. 


2 choose 6 = 15 

[source](https://www.calculatorsoup.com/calculators/discretemathematics/combinations.php)

2. [M] What is the difference between sampling with vs. without replacement? Name an example of when you would use one rather than the other?

sampling with replacement means you pick sample and then put back into sample space. (stateless)

sampling without relacement means sampling and keeping data out of sample space. (stateful)

Of course(!) we use sampling when splitting dataset for train/validation/test

Examples of Sampling without Replacement in Data Science
Sampling without replacement is used throughout data science. One very common use is in model validation procedures like train test split and cross validation. In short, each of these procedures allows you to simulate how a machine learning model would perform on new/unseen data.

The image below shows the train test split procedure which consists of splitting a dataset into two pieces: a training set and a testing set. This consists of randomly sampling WITHOUT replacement about 75% (you can vary this) of the rows and putting them into your training set and putting the remaining 25% to your test set. Note that the colors in “Features” and “Target” indicate where their data will go (“X_train”, “X_test”, “y_train”, “y_test”) for a particular train test split.

[source](https://towardsdatascience.com/understanding-sampling-with-and-without-replacement-python-7aff8f47ebe4)


3. [M] Explain Markov chain Monte Carlo sampling.

todo: learn more about this

Markov chain Monte Carlo methods create samples from a continuous random variable, with probability density proportional to a known function. These samples can be used to evaluate an integral over that variable, as its expected value or variance.

[source](https://www.wikiwand.com/en/Markov_chain_Monte_Carlo)


4. [M] If you need to sample from high-dimensional data, which sampling method would you choose?

Not sure, Reddit says: https://www.wikiwand.com/en/Markov_chain_Monte_Carlo

5. [H] Suppose we have a classification task with many classes. An example is when you have to predict the next word in a sentence -- the next word can be one of many, many possible words. If we have to calculate the probabilities for all classes, it’ll be prohibitively expensive. Instead, we can calculate the probabilities for a small set of candidate classes. This method is called candidate sampling. Name and explain some of the candidate sampling algorithms.

Hint: check out this: https://www.tensorflow.org/extras/candidate_sampling.pdf 

6. Suppose you want to build a model to classify whether a Reddit comment violates the website’s rule. You have 10 million unlabeled comments from 10K users over the last 24 months and you want to label 100K of them.

[M] How would you sample 100K comments to label?
[M] Suppose you get back 100K labeled comments from 20 annotators and you want to look at some labels to estimate the quality of the labels. How many labels would you look at? How would you sample them?

Hint: https://www.cloudresearch.com/resources/guides/sampling/pros-cons-of-different-sampling-methods/ 

7. [M] Suppose you work for a news site that historically has translated only 1% of all its articles. Your coworker argues that we should translate more articles into Chinese because translations help with the readership. On average, your translated articles have twice as many views as your non-translated articles. What might be wrong with this argument?

Hint: think about selection bias.

8. [M] How to determine whether two sets of samples (e.g. train and test splits) come from the same distribution?

https://medium.com/@praveenkotha/how-to-find-whether-train-data-and-test-data-comes-from-same-data-distribution-9259018343b

9. [H] How do you know you’ve collected enough samples to train your ML model?

You can only know by training models and seeing what performance you can get. 

There are rules of thumb though: 
https://towardsdatascience.com/how-do-you-know-you-have-enough-training-data-ad9b1fd679ee

10. [M] How to determine outliers in your data samples? What to do with them?

Plot your datset distribution and look for outliers or programaticaly fit a Gaussian and then look for values that are more than 3 or 4 sigma. 

[how to remove outliers in dataset](https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/)

[scikit-learn outlier detection](https://scikit-learn.org/stable/modules/outlier_detection.html)


11. Sample duplication
* [M] When should you remove duplicate training samples? When shouldn’t you?
* [M] What happens if we accidentally duplicate every data point in your train set or in your test set?


In supervised learning you *should not* remove duplicates because it will change your sample distribution. 

https://www.quora.com/Should-we-remove-duplicates-from-a-data-set-while-training-a-Machine-Learning-algorithm-shallow-and-or-deep-methods


but maybe we should? https://indicodata.ai/blog/should-we-remove-duplicates-ask-slater/

I need to look into this more... 

12. Missing data
* [H] In your dataset, two out of 20 variables have more than 30% missing values. What would you do?
* [M] How might techniques that handle missing data make selection bias worse? How do you handle this bias?


you could try to model those two variables and then see what values you should put in there. Or you could just put the mean in those values. Might try to see those variables are even important anyways. 


"Selection bias is the phenomenon of selecting individuals, groups or data for analysis in such a way that proper randomization is not achieved, ultimately resulting in a sample that is not representative of the population

... 

Handling missing data can make selection bias worse because different methods impact the data in different ways. For example, if you replace null values with the mean of the data, you adding bias in the sense that you’re assuming that the data is not as spread out as it might actually be.
"

13. [M] Why is randomization important when designing experiments (experimental design)?

"Randomization prevents biases and makes the results fair. It makes sure that the groups made for conducting an experiments are as similar as possible to each other so that the results come out as accurate as possible."

14. Class imbalance.
* [E] How would class imbalance affect your model?
* [E] Why is it hard for ML models to perform well on data with class imbalance?
* [M] Imagine you want to build a model to detect skin legions from images. In your training dataset, only 1% of your images shows signs of legions. After training, your model seems to make a lot more false negatives than false positives. What are some of the techniques you'd use to improve your model?






































