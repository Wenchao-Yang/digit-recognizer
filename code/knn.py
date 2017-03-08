# Lisa Zhang
# CSE 446 final project

# A Knn Classifier and a test function which run the classifier with MNIST data


import pandas as pd
import numpy as np
import heapq as hq

# weight = 1  # uniform weight,


class KnnClassifier:
    def __init__(self, train, test, k):
        # self.train = train
        # self.test = test
        self.k = k
        self.train_data = train.drop("label", axis=1)
        self.train_label = train.iloc[0:, 0]
        #print self.train_label
        self.test_data = test
        self.label_frequency_table = pd.DataFrame(data=np.zeros((self.test_data.shape[0], 10)),
                                                  index=range(0, self.test_data.shape[0]),
                                                  columns=range(0, 10))

    # calculate the euclidean distance between vector x and vector y (same dim)
    @staticmethod
    def euclidean_distance(x, y):
        x = np.array(x)
        y = np.array(y)
        distance = np.linalg.norm(x - y, ord=2, axis=0)
        return distance
        # return math.sqrt(sum([(a - b)**2 for a, b in zip(x, y)]))

    # predict the class labels for the provided data
    def predict(self):
        print "Predicting"
        predicted_labels = []
        for i in self.test_data.index:
            target_point = self.test_data.ix[i]  # a row of test data
            # print target_point
            # find k nearest from train
            nearest_k_points = hq.nsmallest(self.k, #enumerate(self.train_data),
                                            self.train_data.itertuples(),
                                            key=lambda point: self.euclidean_distance(target_point, point[1:]))
            # print nearest_k_points
            nearest_labels = [self.train_label.ix[point[0]] for point in nearest_k_points]
            print nearest_labels
            majority_label = max(set(nearest_labels), key=nearest_labels.count)
            predicted_labels.append(majority_label)
            # populate frequency table
            for l in nearest_labels:
                self.label_frequency_table[l][i] += 1
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
    train = pd.read_csv("../data/train.csv")
    test = pd.read_csv("../data/test.csv")
    k = 1

    knn = KnnClassifier(train, test, k)
    predicted_labels = knn.predict()
    print predicted_labels


if __name__ == "__main__":
    test_function()
