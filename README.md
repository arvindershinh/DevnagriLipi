# DevnagriLipi

Description of Project: - This model has been trained to classify 10 different handwritten Devnagri Lipi alphabets (Sanskrit Language Lipi) Images. The model uses Convolution Neural Network for this purpose and has been designed with TensorFlow Low level APIs.

Training Data: - Steps to create training dataset
+ Initially created handwritten alphabets using MS paint software and resized to 28*28 pixels using Python.
+ Used Dataset Augmentation to generate further datasets by translation and rotation of existing Images using self-written python code.
+ Normalization of Images


Architecture of Convolution Neural Network Training Algorithm
+ Training Datasets: - Algorithm shuffles and create fixed size batches of datasets.
+ Model Function: - Model function consist of 6 layers (Convolution+Pooling+Convolution+Pooling+Dense+Dense). Each layer is using ReLU activation function to add non-linearity.
+ Regularization-Used dropout for model regularization. 
+ Output Function: - Used Softmax function as output function to get normalized probability from Model Function Output.
+ Cost Function: - Used Cross Entropy estimator with output of Softmax.
+ Optimization Algorithm: - Used Adam optimizer.

performance measure: - Used Accuracy as performance measure.

Visualization Tool: - Used Tensorboard to visualize loss, Accuracy, weights and biases Histogram graphs and used matplotlib to visualize Image Data.


Deployment (Using Serving and Docker)
+ Using TensorFlow saved model builder APIs, exported model into Protocol buffer format.
+ Creating Model Server
  * Downloaded TensorFlow Serving and installed Docker for window OS
  * From docker terminal, created Docker Image, using script present in TensorFlow Serving.
  * From docker terminal, created Docker Container instance of docker image
  * Started docker container and deploy the exported model Protocol buffer into it.
  * Hosted model at specific port with port forwarding.
+ Creating Model Client
  * compiled serving api protobuf files present in serving folder to Python code using grpcio-tools compiler.
  * Created a Python file and imported above python serving api files in it.
  * In Python File, created Request with server IP address and Port. 
  * Provided input Devnagri lipi image to request and executed python code to execute request and get response.



Self-Learned Machine Learning Concepts 

Before creating machine learning project, I deeply learned the following concepts.

Mathematical Concepts

+ Linear Algebra – Linearly Independent Vectors, Eigendecomposition, Singular Value Decomposition, PCA, Trace Operator, Moore-Penrose Pseudoinverse
+ Probability – i.i.d, central limit theorem, Probability Distributions-(Bernoulli, Gaussian, Laplace, Mixtures of Distributions),
Chain Rule of Conditional Probabilities, Expectation, Variance, Covariance, covariance matrix, [Common Functions-logistic sigmoid, softplus, softmax], Bayes’ Rule, [Information Theory-Kullback-Leibler (KL) divergence, cross-entropy], Structured Probabilistic Models
+ Numerical Computation & Calculus – [Taylor series approximation of a function and its significance], differentiation, integration, 
[Overflow and Underflow], [Poor Conditioning and its challenges], Gradient, Gradient descent, [Jacobian and Hessian Matrices],
[positive and negative definite Matrices], line search, first/second-order optimization algorithms, convex/non-convex optimization problems.


Machine learning Concepts
+ Machine Learning Basics – [Capacity, Overfitting and Underfitting], training error, generalization error, The No Free Lunch Theorem, 
  [Hyperparameters and Validation Sets], k-fold cross-validation algorithm 
  * Frequentist Statistics-[Bias and Variance of Estimators], Standard Error, [relation between Bias, Variance and Mean Squared Error of an Estimator and their link with Overfitting and Underfitting], Consistency, Maximum Likelihood Estimation, [Conditional Log-Likelihood vs Mean Squared Error]
  * Bayesian Statistics-prior/posterior probability distribution, Maximum a Posteriori (MAP) Estimation.
  * Supervised learning algorithms-linear regression, logistic regression, Support Vector Machines, k-nearest neighbors
  * Unsupervised learning algorithms-PCA (lower-dimensional and independent representation of data), k-means Clustering (sparse representation of data).

Deep learning Concepts
+ Deep Feedforward Networks – [output layer Units- Linear/Sigmoid/Softmax, hidden layers], [activation functions- ReLU, other variants of ReLU], 
[Cost Functions- Maximum Likelihood], Chain Rule of Calculus, back-propagation algorithm
+ Regularization Techniques – weight decay (L2, L1 norm), Dataset Augmentation, Noise Robustness, Early Stopping, Semi-Supervised Learning, 
parameter sharing, model averaging, Dropout
+ Optimization – SGD, Momentum, [Adaptive Learning Rates Algorithms- AdaGrad, RMSProp, Adam], Newton’s Method, Batch Normalization technique.

Technologies 
+ Python, Python Frameworks – NumPy, matplotlib, pandas, PIL
+ TensorFlow Framework – TensorFlow Low level APIs, Tensorboard, Serving
+ Docker Container Platform