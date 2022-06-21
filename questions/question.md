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

$$

outer \product:\ {\bf a}\otimes {\bf b}\\ \hspace{50px}{\bf a}\otimes {\bf b} =\normalsize{\left(\begin{array}\\ a_1\\ a_2\\\vdots\\a_i\\\end{array}\right)} \otimes \normalsize{\left(\begin{array}\\ b_1\\ b_2\\\vdots\\b_j\\\end{array}\right)}^{\normalsize t} =\normalsize {\left[\begin{array}\\ a_{\small 1}b_{\small 1}& a_{\small 1}b_{\small 2}& \cdots& a_{\small 1}b_{\small j}\\ a_{\small 2}b_{\small 1}& a_{\small 2}b_{\small 2}& \cdots& a_{\small 2}b_{\small j}\\ \vdots& \vdots& \ddots& \vdots\\ a_{\small i}b_{\small 1}& a_{\small i}b_{\small 2}& \cdots& a_{\small i}b_{\small j}\\\end{array}\right]} =\large {\bf C}\\

$$

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

