# 7.2 Sampling and creating training data

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

Markov chain Monte Carlo methods create samples from a continuous random variable, with probability density proportional to a known function. These samples can be used to evaluate an integral over that variable, as its expected value or variance.

[source](https://www.wikiwand.com/en/Markov_chain_Monte_Carlo)




