# MachineLearningPracticePrograms
This repository contains programs that are often used in machine learning problems implemented with minimal dependencies.
Requirements are python 3.5.3 (or above), numpy, matplotlib(for graphical display).

In this repo we try to implement general machine learning algorithms and calculations for learning purpose. The modules implemented are described below.

### svm ###
A Support Vector Machine (SVM) is a discriminative classifier formally defined by a separating hyperplane. In other words, given labeled training data (supervised learning), the algorithm outputs an optimal hyperplane which categorizes new examples. The calculations performed are based on this [link](https://github.com/MaviccPRP/svm/blob/master/svm-primal.ipynb).

### Simplest Neural Network ###
A neural network is a a computer system modelled on the human brain and nervous system. It is the simplest implementation with toy dataset that one could imagine. Lets begin my neural network journey with this!


### Simple Convolutional Neural Network ###
In machine learning, a convolutional neural network (CNN, or ConvNet) is a class of deep, feed-forward artificial neural network that have successfully been applied to analyzing visual imagery. [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) is implemented using keras over [mnist](http://yann.lecun.com/exdb/mnist/) dataset. Its one of the simplest forms of ConvNet.
(CONV-POOL-CONV-POOL-CONV-FC)

### k-means Clustering ###
k-means clustering aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster. This results in a partitioning of the data space into Voronoi cells. Lloyd's algorithm for k-means clustering is implemented over mnist dataset with varying k from 5 to 25.
