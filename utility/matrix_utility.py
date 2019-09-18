import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


class MatrixUtility:
    def __init__(self, matrix, interacted=None):
        self.matrix = matrix
        self.interacted = interacted
        self.n_x = self.matrix.shape[0]
        self.n_y = self.matrix.shape[1]

    def __compute_mean(self):
        self.mean = np.zeros(self.n_x)
        for x in range(self.n_x):
            r = self.matrix[x]
            if self.interacted is not None:
                r = r[self.interacted[x] == 1]
            self.mean[x] = r.mean() if r.size > 0 else 0.0

    def __compute_pearson(self):
        diff = (self.matrix.T - self.mean).T
        diff[self.interacted == 0] = 0.0
        x = np.dot(diff, diff.T)
        y = np.sqrt(np.dot(diff ** 2, self.interacted.T if self.interacted is not None else np.ones(self.matrix.T.shape)))
        y = y * y.T
        self.similarity = MinMaxScaler().fit_transform(x / (y + 1e-8))

    def __compute_cosine(self):
        self.similarity = cosine_similarity(self.matrix)

    def compute(self):
        self.__compute_mean()
        # self.__compute_pearson()
        self.__compute_cosine()
