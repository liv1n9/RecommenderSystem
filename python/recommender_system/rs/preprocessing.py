import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import coo_matrix


class Data:
    def __init__(self, r_train, r_test):
        self.r_train = r_train.astype(np.float32)
        self.r_test = r_test.astype(np.float32)
        self.matrix = None
        self.rated = None
        self.n_users = None
        self.n_items = None
        self.min_rating = None
        self.max_rating = None
        self.mean_rating = None

    def process(self):
        temp_1 = np.concatenate((self.r_train[:, 0], self.r_test[:, 0]))
        temp_2 = np.concatenate((self.r_train[:, 1], self.r_test[:, 1]))

        le = LabelEncoder()
        le.fit(temp_1)
        self.r_train[:, 0] = le.transform(self.r_train[:, 0])
        self.r_test[:, 0] = le.transform(self.r_test[:, 0])
        le.fit(temp_2)
        self.r_train[:, 1] = le.transform(self.r_train[:, 1])
        self.r_test[:, 1] = le.transform(self.r_test[:, 1])

        self.n_users = int(max(np.max(self.r_train[:, 0]), np.max(self.r_test[:, 0]))) + 1
        self.n_items = int(max(np.max(self.r_train[:, 1]), np.max(self.r_test[:, 1]))) + 1
        self.min_rating = np.min(self.r_train[:, 2])
        self.max_rating = np.max(self.r_train[:, 2])
        self.mean_rating = np.mean(self.r_train[:, 2])

        self.rated = coo_matrix((np.ones(self.r_train.shape[0]), (self.r_train[:, 0], self.r_train[:, 1])),
                                (self.n_users, self.n_items)).toarray()
        self.matrix = coo_matrix((self.r_train[:, 2], (self.r_train[:, 0], self.r_train[:, 1])),
                                 (self.n_users, self.n_items)).toarray()
