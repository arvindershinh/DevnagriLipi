# DevnagriLipi

Description of Project: - This model has been trained to classify 10 different handwritten Devnagri Lipi alphabets (Sanskrit Language Lipi) Images. The model uses Convolution Neural Network for this purpose and has been designed with TensorFlow Low level APIs.

Training Data: - Steps to create training dataset
-> Initially created handwritten alphabets using MS paint software and resized to 28*28 pixels using Python.
-> Used Dataset Augmentation to generate further datasets by translation and rotation of existing Images using self-written python code.
-> Normalization of Images

Architecture of Convolution Neural Network Training Algorithm
-> Training Datasets: - Algorithm shuffles and create fixed size batches of datasets.
-> Model Function: - Model function consist of 6 layers (Convolution?Pooling?Convolution?Pooling?Dense?Dense). Each layer is using ReLU activation function to add non-linearity. 
-> Regularization-Used dropout for model regularization. 
-> Output Function: - Used Softmax function as output function to get normalized probability from Model Function Output.
-> Cost Function: - Used Cross Entropy estimator with output of Softmax.
-> Optimization Algorithm: - Used Adam optimizer.

performance measure: - Used Accuracy as performance measure.

Visualization Tool: - Used Tensorboard to visualize loss, Accuracy, weights and biases Histogram graphs and used matplotlib to visualize Image Data.

Deployment (Using Serving and Docker)
-> Using TensorFlow saved model builder APIs, exported model into Protocol buffer format.
-> Creating Model Server
  * Downloaded TensorFlow Serving and installed Docker for window OS
  * From docker terminal, created Docker Image, using script present in TensorFlow Serving.
  * From docker terminal, created Docker Container instance of docker image
  * Started docker container and deploy the exported model Protocol buffer into it.
  * Hosted model at specific port with port forwarding.
-> Creating Model Client
  * compiled serving api protobuf files present in serving folder to Python code using grpcio-tools compiler.
  * Created a Python file and imported above python serving api files in it.
  * In Python File, created Request with server IP address and Port. 
  * Provided input Devnagri lipi image to request and executed python code to execute request and get response.
