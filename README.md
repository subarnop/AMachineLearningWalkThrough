# MachineLearningPracticePrograms
This repository contains algorithms and techniques that are often used in machine learning problems implemented with minimal dependencies possible.
Requirements are python 3.5.3 (or above), keras, numpy, matplotlib(for graphical display).

In this repo we try to implement general machine learning algorithms and calculations for learning purpose. The modules implemented are described below.

### SVM ###
A Support Vector Machine (SVM) is a discriminative classifier formally defined by a separating hyperplane. In other words, given labeled training data (supervised learning), the algorithm outputs an optimal hyperplane which categorizes new examples. The calculations performed are based on this [link](https://github.com/MaviccPRP/svm/blob/master/svm-primal.ipynb).

### Simplest Neural Network ###
A neural network is a a computer system modelled on the human brain and nervous system. It is the simplest implementation with toy dataset that one could imagine. Lets begin my neural network journey with this!


### Simple Convolutional Neural Network ###
In machine learning, a convolutional neural network (CNN, or ConvNet) is a class of deep, feed-forward artificial neural network that have successfully been applied to analyzing visual imagery. [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) is implemented using keras over [mnist](http://yann.lecun.com/exdb/mnist/) dataset. Its one of the simplest forms of ConvNet.
(CONV-POOL-CONV-POOL-CONV-FC)

### k-means Clustering ###
k-means clustering aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster. This results in a partitioning of the data space into Voronoi cells. Lloyd's algorithm for k-means clustering is implemented over mnist dataset with varying k from 5 to 25.

### Recurrent Neural Network ###
A recurrent neural network (RNN) is a class of artificial neural network where connections between units form a directed cycle. This allows it to exhibit dynamic temporal behavior. Unlike feedforward neural networks, RNNs can use their internal memory to process arbitrary sequences of inputs. We develop a word level Vanilla RNN for predicting sequences of text.

### Logistic Regression ###
 Logistic regression is a generalized linear model that we can use to model or predict categorical outcome variables. In logistic regression, we’re essentially trying to find the weights that maximize the likelihood of producing our given data and use them to categorize the response variable. We implement a binary classifier in Logistic Regression model that classifies a toy data built on Gaussian distribution.

 ### Naive Bayesian Classifier ###
  It is a classification technique based on Bayes' Theorem with an assumption of independence among predictors. In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature. We implement a binary classifier in Naive Bayesian model that classifies over [pima-indians-diabetes dataset](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes) from UCI Machine learning repository.
