import numpy as np
import matplotlib.pyplot as plt
import random

from base64 import b64decode
from json import loads

def parse(x):
    #parse the digits file into tuples of
    digit = loads(x)
    array = np.fromstring(b64decode(digit["data"]),dtype=np.ubyte)
    array = array.astype(np.float64)
    return (digit["label"], array)

def cluster(labelled_data, k):
    centroids = init_centroids(labelled_data, k)
    clusters = form_clusters(labelled_data, centroids)
    final_clusters, final_centroids = repeat_until_convergence(labelled_data, clusters, centroids)
    return final_clusters, final_centroids

def init_centroids(labelled_data,k):
    #Initialize random centroid positions
    return list(map(lambda x: x[1], random.sample(labelled_data,k)))

def form_clusters(labelled_data, unlabelled_centroids):
    #allocate each datapoint to its closest centroid
    centroids_indices = range(len(unlabelled_centroids))
    clusters = {c: [] for c in centroids_indices}
    for (label,Xi) in labelled_data:
        # for each datapoint, pick the closest centroid.
        smallest_distance = float("inf")
        for cj_index in centroids_indices:
            cj = unlabelled_centroids[cj_index]
            distance = np.linalg.norm(Xi - cj)
            if distance < smallest_distance:
                closest_centroid_index = cj_index
                smallest_distance = distance
        # allocate that datapoint to the cluster of that centroid.
        clusters[closest_centroid_index].append((label,Xi))
    return clusters.values()

def repeat_until_convergence(labelled_data, labelled_clusters, unlabelled_centroids):
    #find best fitting centroids to the labelled_data
    previous_max_difference = 0
    while True:
        unlabelled_old_centroids = unlabelled_centroids
        unlabelled_centroids = move_centroids(labelled_clusters)
        labelled_clusters = form_clusters(labelled_data, unlabelled_centroids)

        differences = list(map(lambda a, b: np.linalg.norm(a-b),unlabelled_old_centroids,unlabelled_centroids))
        max_difference = max(differences)
        if np.isnan(max_difference-previous_max_difference):
            difference_change = np.nan
        else:
            difference_change = abs((max_difference-previous_max_difference)/np.mean([previous_max_difference,max_difference])) * 100

        previous_max_difference = max_difference
        # difference change is nan once the list of differences is all zeroes.
        if np.isnan(difference_change):
            break
    return labelled_clusters, unlabelled_centroids

def move_centroids(labelled_clusters):
    """
    returns a list of centroids corresponding to the clusters.
    """
    new_centroids = []
    for cluster in labelled_clusters:
        new_centroids.append(mean_cluster(cluster))
    return new_centroids

def mean_cluster(labelled_cluster):
    #return mean of a labelled_cluster
    sum_of_points = sum_cluster(labelled_cluster)
    mean_of_points = sum_of_points * (1.0 / len(labelled_cluster))
    return mean_of_points

def sum_cluster(labelled_cluster):
    # assumes len(cluster) > 0
    sum_ = labelled_cluster[0][1].copy()
    for (label,vector) in labelled_cluster[1:]:
        sum_ += vector
    return sum_

def assign_labels_to_centroids(clusters, centroids):
    #assign a label to each centroid
    labelled_centroids = []
    clusters = list(clusters)

    for i in range(len(clusters)):
        labels = list(map(lambda x: x[0], clusters[i]))
        # pick the most common label
        most_common = max(set(labels), key=labels.count)
        centroid = (most_common, centroids[i])
        labelled_centroids.append(centroid)

    return labelled_centroids

def get_error_rate(digits,labelled_centroids):
    #classifies a list of labelled digits. returns the error rate.
    classified_incorrect = 0
    for (label,digit) in digits:
        classified_label = classify_digit(digit, labelled_centroids)
        if classified_label != label:
            classified_incorrect +=1
    error_rate = classified_incorrect / float(len(digits))
    return error_rate
    
def classify_digit(digit, labelled_centroids):
    mindistance = float("inf")
    for (label, centroid) in labelled_centroids:
        distance = np.linalg.norm(centroid - digit)
        if distance < mindistance:
            mindistance = distance
            closest_centroid_label = label
    return closest_centroid_label

#Read the data file
with open("data/digits.base64.json","r") as f:
    digits = list(map(parse, f.readlines()))

#devide the dataset into training and validation set
ratio = int(len(digits)*0.25)
validation = digits[:ratio]
training = digits[ratio:]

error_rates = {x:None for x in range(5,25)}
for k in range(5,25):
    #training
    trained_clusters, trained_centroids = cluster(training, k)
    labelled_centroids = assign_labels_to_centroids(trained_clusters, trained_centroids)
    #validation
    error_rate = get_error_rate(validation, labelled_centroids)
    error_rates[k] = error_rate

# Show the error rates
x_axis = sorted(error_rates.keys())
y_axis = [error_rates[key] for key in x_axis]
plt.figure()
plt.title("Error Rate by Number of Clusters")
plt.scatter(x_axis, y_axis)
plt.xlabel("Number of Clusters")
plt.ylabel("Error Rate")
plt.savefig('Results/kmeans_acc.png')
