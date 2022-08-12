# ML interview Questions

https://huyenchip.com/ml-interviews-book/contents/5.1.1-vectors.html

# Vectors 

1.1 What is the geometric interpretation of the dot product? 


Dot product is 

$$ 
a \dot b = ||a|| ||b||  \cos{\theta}
$$

So the geometric interpretation is the angle between two vectors. 

When vectors are orthogonal $ \cos{90} = 1$ and when they are in the same direction $ \cos{0} = 0$ 

1.2 Given a vector $u$, find vector $v$ of unit length such that the dot product of $u$ and $v$ is maximum


2.2 Given two vectors $a=[3,2,1]$ and $b=[−1,0,1]$. Calculate the outer product $a^{T}b$?

returns a matrix of $\mathbb{R}^{n x n}$

The general form of an outer product is:

$ u = \begin{bmatrix}
           u_{1} \\
           u_{2} \\
           \vdots \\
           u_{m}
         \end{bmatrix}
$

$ v = \begin{bmatrix}
           v_{1} \\
           v_{2} \\
           \vdots \\
           v_{n}
         \end{bmatrix}
$

$ u \otimes  v = \begin{bmatrix} 
    u_1 v_1 &  u_1 v_2 & \dots & u_1 v_n \\
    u_2 v_2 &  u_2 v_2 & \dots & u_2 v_n \\
    \vdots &  \vdots & \ddots & \vdots \\
    u_m v_1 &  u_m v_2 & \dots & u_m v_n \\
 \end{bmatrix}
$


2.2 Give an example of how the outer product can be useful in ML.

# TODO
Error backprop? 


3. What does it mean for two vectors to be linearly independent?

In the theory of vector spaces, a set of vectors is said to be linearly dependent if there is a nontrivial linear combination of the vectors that equals the zero vector. If no such linear combination exists, then the vectors are said to be linearly independent. These concepts are central to the definition of dimension.

4. Given two sets of vectors $A=a_1,a_2,a_3,...,a_n$ and $B=b_1,b_2,b_3,...,b_m$. How do you check that they share the same basis?

you would have to check that they are multiples of one another. 

5. Given n vectors, each of d dimensions. What is the dimension of their span?

You would have to row reduce to see how many dimensions you're left with! 

6.1 What's a norm? $L_1, L_2, L_{norm}$?



6.2 How do norm and metric differ? Given a norm, make a metric. Given a metric, can we make a norm?


# Probability 


[E] Given a uniform random variable in the range of inclusively. What’s the probability that ?

[E] Can the values of PDF be greater than 1? If so, how do we interpret PDF?

[E] What’s the difference between multivariate distribution and multimodal distribution?

[E] What does it mean for two variables to be independent?

[E] It’s a common practice to assume an unknown variable to be of the normal distribution. Why is that?

[E] How would you turn a probabilistic model into a deterministic model?

[H] Is it possible to transform non-normal variables into normal variables? How?

[M] When is the t-distribution useful?

### Assume you manage an unreliable file storage system that crashed 5 times in the last year, each crash happens independently.

[M] What's the probability that it will crash in the next month?

[M] What's the probability that it will crash at any given moment?

[M] Say you built a classifier to predict the outcome of football matches. In the past, it's made 10 wrong predictions out of 100. Assume all predictions are made independently., what's the probability that the next 20 predictions are all correct?

[M] Given two random variables and . We have the values and for all values of and . How would you calculate ?

[M] You know that your colleague Jason has two children and one of them is a boy. What’s the probability that Jason has two sons? Hint: it’s not .

### There are only two electronic chip manufacturers: A and B, both manufacture the same amount of chips. A makes defective chips with a probability of 30%, while B makes defective chips with a probability of 70%.

* [E] If you randomly pick a chip from the store, what is the probability that it is defective?


* [M] Suppose you now get two chips coming from the same company, but you don’t know which one. When you test the first chip, it appears to be functioning. What is the probability that the second electronic chip is also good?


### There’s a rare disease that only 1 in 10000 people get. Scientists have developed a test to diagnose the disease with the false positive rate and false negative rate of 1%.

    [E] Given a person is diagnosed positive, what’s the probability that this person actually has the disease?
    [M] What’s the probability that a person has the disease if two independent tests both come back positive?

[M] A dating site allows users to select 10 out of 50 adjectives to describe themselves. Two users are said to match if they share at least 5 adjectives. If Jack and Jin randomly pick adjectives, what is the probability that they match?
[M] Consider a person A whose sex we don’t know. We know that for the general human height, there are two distributions: the height of males follows and the height of females follows . Derive a probability density function to describe A’s height.

[H] There are three weather apps, each the probability of being wrong ⅓ of the time. What’s the probability that it will be foggy in San Francisco tomorrow if all the apps predict that it’s going to be foggy in San Francisco tomorrow and during this time of the year, San Francisco is foggy 50% of the time?

Hint: you’d need to consider both the cases where all the apps are independent and where they are dependent.
[M] Given samples from a uniform distribution . How do you estimate ? (Also known as the German tank problem)
[M] You’re drawing from a random variable that is normally distributed, , once per day. What is the expected number of days that it takes to draw a value that’s higher than 0.5?
[M] You’re part of a class. How big the class has to be for the probability of at least a person sharing the same birthday with you is greater than 50%?
[H] You decide to fly to Vegas for a weekend. You pick a table that doesn’t have a bet limit, and for each game, you have the probability of winning, which doubles your bet, and of losing your bet. Assume that you have unlimited money (e.g. you bought Bitcoin when it was 10 cents), is there a betting strategy that has a guaranteed positive payout, regardless of the value of ?
[H] Given a fair coin, what’s the number of flips you have to do to get two consecutive heads?

[H] In national health research in the US, the results show that the top 3 cities with the lowest rate of kidney failure are cities with populations under 5,000. Doctors originally thought that there must be something special about small town diets, but when they looked at the top 3 cities with the highest rate of kidney failure, they are also very small cities. What might be a probabilistic explanation for this phenomenon?

Hint: The law of small numbers.
[M] Derive the maximum likelihood estimator of an exponential distribution.


# Statistics 




[E] Explain frequentist vs. Bayesian statistics.
[E] Given the array , find its mean, median, variance, and standard deviation.
[M] When should we use median instead of mean? When should we use mean instead of median?
[M] What is a moment of function? Explain the meanings of the zeroth to fourth moments.
[M] Are independence and zero covariance the same? Give a counterexample if not.
[E] Suppose that you take 100 random newborn puppies and determine that the average weight is 1 pound with the population standard deviation of 0.12 pounds. Assuming the weight of newborn puppies follows a normal distribution, calculate the 95% confidence interval for the average weight of all newborn puppies.

[M] Suppose that we examine 100 newborn puppies and the 95% confidence interval for their average weight is pounds. Which of the following statements is true?
    Given a random newborn puppy, its weight has a 95% chance of being between 0.9 and 1.1 pounds.
    If we examine another 100 newborn puppies, their mean has a 95% chance of being in that interval.

    We're 95% confident that this interval captured the true mean weight.

    Hint: This is a subtle point that many people misunderstand. If you struggle with the answer, Khan Academy has a great article on it.
[H] Suppose we have a random variable supported on from which we can draw samples. How can we come up with an unbiased estimate of the median of ?
[H] Can correlation be greater than 1? Why or why not? How to interpret a correlation value of 0.3?
The weight of newborn puppies is roughly symmetric with a mean of 1 pound and a standard deviation of 0.12. Your favorite newborn puppy weighs 1.1 pounds.
    [E] Calculate your puppy’s z-score (standard score).
    [E] How much does your newborn puppy have to weigh to be in the top 10% in terms of weight?
    [M] Suppose the weight of newborn puppies followed a skew distribution. Would it still make sense to calculate z-scores?
[H] Tossing a coin ten times resulted in 10 heads and 5 tails. How would you analyze whether a coin is fair?
Statistical significance.
    [E] How do you assess the statistical significance of a pattern whether it is a meaningful pattern or just by chance?
    [E] What’s the distribution of p-values?
    [H] Recently, a lot of scientists started a war against statistical significance. What do we need to keep in mind when using p-value and statistical significance?
Variable correlation.
    [M] What happens to a regression model if two of their supposedly independent variables are strongly correlated?
    [M] How do we test for independence between two categorical variables?
    [H] How do we test for independence between two continuous variables?
[E] A/B testing is a method of comparing two versions of a solution against each other to determine which one performs better. What are some of the pros and cons of A/B testing?
[M] You want to test which of the two ad placements on your website is better. How many visitors and/or how many times each ad is clicked do we need so that we can be 95% sure that one placement is better?
[M] Your company runs a social network whose revenue comes from showing ads in newsfeed. To double revenue, your coworker suggests that you should just double the number of ads shown. Is that a good idea? How do you find out?

Imagine that you have the prices of 10,000 stocks over the last 24 month period and you only have the price at the end of each month, which means you have 24 price points for each stock. After calculating the correlations of 10,000 * 9,9992 pairs of stock, you found a pair that has the correlation to be above 0.8.
    [E] What’s the probability that this happens by chance?
    [M] How to avoid this kind of accidental patterns?

Hint: Check out the curse of big data.
[H] How are sufficient statistics and Information Bottleneck Principle used in machine learning?

