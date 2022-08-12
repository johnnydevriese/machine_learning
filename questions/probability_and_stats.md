##### 5.2.1.2 Questions

From Chip Huyen [ML interviews](https://github.com/chiphuyen/ml-interviews-book/tree/master/contents)

>1. [E] Given a uniform random variable $$X$$ in the range of $$[0, 1]$$ inclusively. What’s the probability that $$X=0.5$$?

zero maybe because we need to integrate between two points. 

For a continuous random variable, the probability that it takes a specific value is zero. So in your case

[source](https://stats.stackexchange.com/questions/142544/probability-from-normal-distribution-vs)

>2. [E] Can the values of PDF be greater than 1? If so, how do we interpret PDF?

Yes, It is the integral of the pdf that must sum to 1. 

>3. [E] What’s the difference between multivariate distribution and multimodal distribution?

Multivariate means there are multiple variables. e.g. x1, x2, x3. 

multimodal means that a variable has more than one mode, or multiple humps. 

[source](https://stats.stackexchange.com/questions/168586/what-is-the-difference-between-multimodal-and-multivariate)

>4. [E] What does it mean for two variables to be independent?

Independence is a fundamental notion in probability theory, as in statistics and the theory of stochastic processes. Two events are independent, statistically independent, or stochastically independent[1] if, informally speaking, the occurrence of one does not affect the probability of occurrence of the other or, equivalently, does not affect the odds. Similarly, two random variables are independent if the realization of one does not affect the probability distribution of the other. 

[wiki](https://www.wikiwand.com/en/Independence_(probability_theory))

>5. [E] It’s a common practice to assume an unknown variable to be of the normal distribution. Why is that?

because the central limit says that iid variables will be gaussian. K Murphy also points out that it's popular because it's easy to compute derivatives with. (Find page in PML for this)

> 6. [E] How would you turn a probabilistic model into a deterministic model?

A deterministic mathematical model is meant to yield a single solution describing the outcome of some "experiment" given appropriate inputs. A probabilistic model is, instead, meant to give a distribution of possible outcomes (i.e. it describes all outcomes and gives some measure of how likely each is to occur).

[source](https://www.quora.com/What-is-the-difference-between-probabilistic-and-deterministic-models)


> 7. [H] Is it possible to transform non-normal variables into normal variables? How?

Yes, using some of the methods in this write up:

https://aegis4048.github.io/transforming-non-normal-distribution-to-normal-distribution


> 8. [M] When is the t-distribution useful?

The t-distribution is a type of normal distribution that is used for smaller sample sizes. Normally-distributed data form a bell shape when plotted on a graph, with more observations near the mean and fewer observations in the tails.

[source](https://www.scribbr.com/statistics/t-distribution/#:~:text=The%20t%20-distribution%20is%20used%20when%20data%20are,data%20set%20%28total%20number%20of%20observations%20minus%201%29.)

> 9. Assume you manage an unreliable file storage system that crashed 5 times in the last year, each crash happens independently.
	1. [M] What's the probability that it will crash in the next month?
	1. [M] What's the probability that it will crash at any given moment?


> 10. [M] Say you built a classifier to predict the outcome of football matches. In the past, it's made 10 wrong predictions out of 100. Assume all predictions are made independently, what's the probability that the next 20 predictions are all correct?

10/100 = 10% wrong -> 90% right 

binomial distribution calculation 

https://math.stackexchange.com/questions/1112657/what-is-the-probability-of-rain-over-a-number-of-days-given-the-probability-of


11. [M] Given two random variables $$X$$ and $$Y$$. We have the values $$P(X|Y)$$ and $$P(Y)$$ for all values of $$X$$ and $$Y$$. How would you calculate $$P(X)$$?

probably have to marginalize out probability 

$P(X) = $

12. [M] You know that your colleague Jason has two children and one of them is a boy. What’s the probability that Jason has two sons? Hint: it’s not $$\frac{1}{2}$$.
13. There are only two electronic chip manufacturers: A and B, both manufacture the same amount of chips. A makes defective chips with a probability of 30%, while B makes defective chips with a probability of 70%.
	1. [E] If you randomly pick a chip from the store, what is the probability that it is defective?
	1. [M] Suppose you now get two chips coming from the same company, but you don’t know which one. When you test the first chip, it appears to be functioning. What is the probability that the second electronic chip is also good?
14. There’s a rare disease that only 1 in 10000 people get. Scientists have developed a test to diagnose the disease with the false positive rate and false negative rate of 1%.
	1. [E] Given a person is diagnosed positive, what’s the probability that this person actually has the disease?
	1. [M] What’s the probability that a person has the disease if two independent tests both come back positive?
15. [M] A dating site allows users to select 10 out of 50 adjectives to describe themselves. Two users are said to match if they share at least 5 adjectives. If Jack and Jin randomly pick adjectives, what is the probability that they match?
16. [M] Consider a person A whose sex we don’t know. We know that for the general human height, there are two distributions: the height of males follows $$h_m = N(\mu_m, \sigma_m^2)$$ and the height of females follows $$h_j = N(\mu_j, \sigma_j^2)$$ . Derive a probability density function to describe A’s height.
17. [H] There are three weather apps, each the probability of being wrong ⅓ of the time. What’s the probability that it will be foggy in San Francisco tomorrow if all the apps predict that it’s going to be foggy in San Francisco tomorrow and during this time of the year, San Francisco is foggy 50% of the time?
	
	**Hint**: you’d need to consider both the cases where all the apps are independent and where they are dependent.
18. [M] Given $$n$$ samples from a uniform distribution $$[0, d]$$. How do you estimate $$d$$? (Also known as the German tank problem)
19. [M] You’re drawing from a random variable that is normally distributed, $$X \sim N(0,1)$$, once per day. What is the expected number of days that it takes to draw a value that’s higher than 0.5?
20. [M] You’re part of a class. How big the class has to be for the probability of at least a person sharing the same birthday with you is greater than 50%?
21. [H] You decide to fly to Vegas for a weekend. You pick a table that doesn’t have a bet limit, and for each game, you have the probability $$p$$ of winning, which doubles your bet, and $$1-p$$ of losing your bet. Assume that you have unlimited money (e.g.  you bought Bitcoin when it was 10 cents), is there a betting strategy that has a guaranteed positive payout, regardless of the value of $$p$$?
22. [H] Given a fair coin, what’s the number of flips you have to do to get two consecutive heads?
23. [H] In national health research in the US, the results show that the top 3 cities with the lowest rate of kidney failure are cities with populations under 5,000. Doctors originally thought that there must be something special about small town diets, but when they looked at the top 3 cities with the highest rate of kidney failure, they are also very small cities. What might be a probabilistic explanation for this phenomenon?

	**Hint**: The law of small numbers.
24. [M] Derive the maximum likelihood estimator of an exponential distribution.

---
*This book was created by [Chip Huyen](https://huyenchip.com) with the help of wonderful friends. For feedback, errata, and suggestions, the author can be reached [here](https://huyenchip.com/communication/). Copyright ©2021 Chip Huyen.*
