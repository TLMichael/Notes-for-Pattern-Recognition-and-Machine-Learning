# Chapter 2　Probability Distributions

- **density estimation**: To model the probability distribution p(x) of a random variable x, given a finite set x1, ..., xn of observations.
- **parametric distributions**: Binomial and multinomial distributions for discrete random variables and the Gaussian distribution for continuous random variables.
- **nonparametric density estimation**: These models contain parameters, which controls the model complexity rather than the form of the distribution. Such as histograms, nearest-neighbours, and kernels.


# Chapter 3　Linear Models for Regression

- **basis function**: The simplest form of linear regression models are also linear functions of the input variables. However, we can obtain a much more useful class of funtions by taking linear combinations of a fixed set of nonlinear functions of input variables, known as *basis functions*.
- **linear models**: Linear models have significant limitations as practical techniques for pattern recognition, particularly for problems involving input spaces of high dimensionality, but they have nice analytical properties and form the foundation for more sophisticated models to be dicussed in later chapters.
- **some basis functions**: Polynomials, spline functions, the sigmoidal basis function, the logistic sigmoid function, the 'tanh' function, the Fourier basis function, wavelets.
- **singular value decomposition**: In linear models, when two or more of the basis vectors are co-linear, or nearly so, the resulting parameter values can have large magnitudes. Such near degeneracies will be very common when dealing with real data sets. The resulting numerical difficulties can be addressed using the technique of SVD.
- **geometrical interpretation of the least-squares solution**: The least-squares regression function is obtained by finding the orthogonal projection of the data vector **t** onto the subspace spanned by the basis function in which each basis function is viewed as a vector of length N.
- **sequential learning**: Sequential algorithms, alse known as on-line algorithms, in which the data points are considered one at a time, and the model parameters updeted after each such presentation. Sequential learning is also appropriate for real-time applications in which the data observations are arriving in a continuous stream, and predictions must be made before all of the data points are seen.
- **bias and variance**: The expected loss can be decomposed into sum of a bias, a variance, and a constant noise term. There is a trade-off between bias and variance. The model with the optimal predictive capability is the one that leads to the best balance between bias and variance.
- **Bayesian treatment of linear regression**: It will avoid over-fitting problem of maximum likelihood, and also lead to automatic methods of determining model complexity using the training data alone.
- **parameter distribution**: A prior probability distribution over the model parameters **w**.
- **predictive distribution**: In practice, we are not usually interested in the value of **w** itself but rather in making predictions of *t* for new values of **x**. This requires that we evaluate the *predictive distribution*.
- **Bayesian model comparison**: The Bayesian view of model comparisan simply involves the use of probabilies to represent uncertainty in the choice of model. Giving the training data ***D***, we then wish to evaluate the posterior distribution *p*(***M*** | ***D***) ∝ *p*(***M***) *p*(***D*** | ***M***).
- **Bayesian approach**: Like any approach to pattern recognition, it needs to make sumptions about the form of the model, and if these are invalid then the results can be misleading.
- **evidence approximation**: Here we discuss an approximation in which we set the hyperparameters to specific values determined by maximizing the *marginal likelihood function* obtained by first integrating over the parameters **w**. This framework is known in the statistics literature as *empirical Bayes*, or *type 2 maximum likelihood*, or *generalized maximum likelihood*, and in the machine learning literature is alse called the *evidance approximation*.
- **the maximization of the log evidence**: There are two approaches: 1) Evaluate the evidence function analytically and then set its derivative equal to zero to obtain re-estimation equations for α and β. 2) Expectation maximization.
- **limitations of fixed basis functions**: The difficulty stems from the assumption that the basis functions are fixed before the training data set is observed and is a manifestation of the curse of dimensionality. As a consequence, the number of basis functions needs to grow rapidly, often exponentially, with the dimensionalty *D* of the input space.
- **alleviation of the last problem**: There are two properties of real data sets. First of all, the data vectors typically lie close to a nonlinear manifold whose intrinsic dimensionality is smaller than that of the input space as a result of strong correlations between the input variables. The second property is that target variables may have significant dependence on only a small number of possible directions within the data manifold. Neural networks can exploit this property by choosing the directions in input space to which the basis functions respond.


# Chapter 4　Linear Models for Classification

- **linearly separable**: Data sets whose classes can be separated exactly by linear decision surfaces.
- **three approaches to the classification problem**: The simplest involves constructing a *discriminant function* that directly assigns each vector **x** to a specific class. A more powerful approach, however, models the conditional probability distribution *p*(***C****k* ｜**x**) in an inference stage, and then subsequently uses this distribution to make optimal decisions. There are two different approaches to determining the conditional probabilities. One technique is to model them directly, for example by representing them as parametric models and then optimizing the parameters using a training set. Alternatively, we can adopt a generative approach in which we model the class-conditional densities given by  *p*(**x**｜***C****k* ), together with the prior probabilities p(***C****k*) for the classes, and we compute the required posterior probabilities using Bayes' theorem.
- **two classes**: The decision boundary corresponds to a (*D* - 1)-dimensinal hyperplane within the *D*-dimensional input space.
- **multiple classes**: 1) *one-versus-the-rest* classifier: Consider the use of K - 1 classifiers each of which soves a two-class problem of separating points in a particular class ***C****k* from points not in the class. 2) *one-versus-one* classifier: K(K - 1)/2 binary discriminant functions, one for every possible pair of classes. 3) Considering a single K-class discriminant comprising K linear functions of the form in two classes classifier.
- **Fisher's linear discriminant**: The Fisher criterion is defined to be the ratio of the between-class variance to the with-in variance. If we adopt a slightly different target coding scheme, then the least-squares solutionfor the weights becomes equivalent to the Fisher solution.
- **the perceptron algorithm**: Applying the stochastic gradient descent algorithm to the *perceptron criterion*, the *perceptron convergence theorem* states that if there exists an exact solution (in other words, if the training data set is linearly separable), then the perceptron learning algorithm is guaranteed to find an exact solution in a finite number of steps.
- **two important activation functions**: *logistic sigmoid function*, *softmax function*.
- **iterative reweighted least squares**: Use the functional form of the generalized linear model explicitly and determine its parameters directly by using maximum likelihood.
- **Laplace approximation**: The Laplace approximation aims to find a Guassian approximation to a probability density defined over a set of continuous variables.


# Chapter 4　Neural Networks

- **the feed-forward neural network**: Also known as the *multilayer perceptron*, which aims to fix the number of basis functions in advance but allow them to be adaptive, in other words to use parametric forms for the basis functions in which the parameter values are adapted during training.
