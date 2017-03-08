# Lisa Zhang
# CSE 446 final project

# A Knn Classifier and a test function which run the classifier with MNIST data


import pandas as pd
import numpy as np
import heapq as hq
from scipy.stats import itemfreq

# weight = 1  # uniform weight


class KnnClassifier:
    def __init__(self, train, test, k):
        # self.train = train
        # self.test = test
        self.k = k
        self.train_data = train.drop("label", axis=1).values
        self.train_label = train.iloc[0:, 0].values
        self.test_data = test.values
        # self.label_frequency_table = np.zeros((self.test_data.shape[0], 10))

    # predict the class labels for the provided data
    def predict(self):
        print "Predicting"
        predicted_labels = np.zeros(len(self.test_data))

        for i in range(0, len(self.test_data)):
            # calculate the distance between this target point and all train data
            dist = np.linalg.norm(self.train_data - self.test_data[i], axis=1, ord=2)

            # find k smallest distance from train. This outputs a list of (index, distance)
            smallest_k_distances_index_pair = hq.nsmallest(self.k, enumerate(dist), key=lambda d: d[1])

            # extract the labels
            nearest_labels = [self.train_label[pair[0]] for pair in smallest_k_distances_index_pair]
            majority_label = max(set(nearest_labels), key=nearest_labels.count)
            predicted_labels[i] = majority_label
            print majority_label

            # populate frequency table
            # for l in nearest_labels:
            #     self.label_frequency_table[l][i] += 1
        return predicted_labels

    # calculate the probability table
    def predict_prob(self):
        return self.label_frequency_table

"""
for each test data, find k nearest neighbor for it from training data, and
predict based on majority voting. Use the euclidean distance for finding
closest points.
"""


def test_function():
    train = pd.read_csv("../data/train.csv", dtype=pd.np.float32)
    test = pd.read_csv("../data/test.csv", dtype=pd.np.float32)
    k = 3
    knn = KnnClassifier(train, test, k)
    predicted_labels = knn.predict()
    print predicted_labels


if __name__ == "__main__":
    test_function()
