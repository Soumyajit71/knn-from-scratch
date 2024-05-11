import numpy as np

import numpy as np
from collections import Counter

def calculate_distance(point_a, point_b):
    return np.linalg.norm(point_a - point_b)


class Knn:
    def __init__(self, k=5):
        self.n_neighbours = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred=[]
        for i in X_test:
            # calculate the distance from each training point
            distances = []
            for j in self.X_train:
                distances.append(calculate_distance(i, j))
            n_neighbours=sorted(list(enumerate(distances)),key= lambda x:x[1])[0:self.n_neighbours]
            label=self.mejority_count(n_neighbours)
            y_pred.append(label)
        return np.array(y_pred)
    def mejority_count(self,neighbours):
        votes=[]
        for i in neighbours:
            votes.append(self.y_train[i[0]])
        votes=Counter(votes)
        return votes.most_common()[0][0]

