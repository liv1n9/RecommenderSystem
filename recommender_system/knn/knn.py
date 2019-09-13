import numpy as np
from utility.matrix_utility import MatrixUtility

DTYPE_INT = np.int32
DTYPE_DOUBLE = np.float64


class KNN:
    def __init__(self, data, user_knn=True, a0=14, c0=0.5, c1=1, d0=10, content_base=False, a1=0.75, b0=0, b1=0.8):
        self.user_knn = user_knn
        if self.user_knn:
            self.rating = data.rating
            self.rated = data.rated
            self.attr = data.attr_items
        else:
            self.rating = data.rating.T
            self.rated = data.rated.T
            self.attr = data.attr_users
        self.n = self.rating.shape[0]
        self.min_rating = data.min_rating
        self.max_rating = data.max_rating

        self.a0 = a0
        self.c0 = c0
        self.c1 = int(c1 * self.n)
        self.d0 = d0

        self.content_base = content_base

        self.a1 = int(a1 * self.attr.shape[1])
        self.b0 = b0
        self.b1 = b1

    def __compute_utility(self):
        matrix_obj_1 = MatrixUtility(self.rating, self.rated)
        matrix_obj_1.compute()
        if not self.content_base:
            self.mean = matrix_obj_1.mean
            self.similarity = matrix_obj_1.similarity
        else:
            dx = np.dot(self.rating, self.attr)
            dy = np.dot(self.rated, self.attr)
            t = dx.mean() * 2.0 / 3.0
            dy[dy < t] = t
            self.content = dx / dy
            matrix_obj_2 = MatrixUtility(self.content)
            e_matrix = np.concatenate((self.rating, self.content), axis=1)
            e_interacted = np.concatenate((self.rated, np.ones(self.content.shape)), axis=1)
            matrix_obj_3 = MatrixUtility(e_matrix, e_interacted)

            matrix_obj_2.compute()
            matrix_obj_3.compute()
            matrix_obj_3.similarity[matrix_obj_1.similarity < self.b0] = -1.0
            matrix_obj_3.similarity[matrix_obj_2.similarity < self.b1] = -1.0

            self.mean = matrix_obj_1.mean
            self.similarity = matrix_obj_3.similarity

    def __compute_neighbor(self):
        self.neighbor = [None] * self.n
        c = np.dot(self.rated, self.rated.T)
        if self.content_base:
            b = np.where(self.content > 0.0, 1.0, 0.0)
            d = np.dot(b, b.T)
        for x in range(self.n):
            y = np.where(c[x] >= self.a0)[0].astype(DTYPE_INT)
            y = y[y != x]
            if self.content_base:
                y = y[d[x, y] >= self.a1]
            sim = self.similarity[x, y]
            y = y[sim >= self.c0]
            sim = self.similarity[x, y]
            ids = np.argsort(sim)[-self.c1:]
            self.neighbor[x] = y[ids]

    def compute(self):
        self.__compute_utility()
        self.__compute_neighbor()

    def predict_group(self, x):
        neighbor = self.neighbor[x]
        if neighbor.size == 0:
            return np.array([]), np.array([])
        mask = np.sum(self.rated[neighbor], axis=0)
        mask[self.rated[x] == 1] = 0
        y = np.where(mask >= self.d0)[0]

        s = self.similarity[x, neighbor]
        rated_y = self.rated[neighbor][:, y]
        rating_y = (self.rating[neighbor][:, y].T - self.mean[neighbor]).T
        rating_y[rated_y == 0] = 0.0

        p = np.dot(s, rating_y) / (np.dot(np.abs(s), rated_y) + 1e-8) + self.mean[x]
        p[p < self.min_rating] = self.min_rating
        p[p > self.max_rating] = self.max_rating

        return y, p

    def __predict(self, x, y):
        m = self.mean[x]
        neighbor = self.neighbor[x]
        r = self.rating[neighbor, y] - m
        s = self.similarity[x, neighbor]
        s[self.rated[neighbor, y] == 0] = 0
        p = (r * s).sum() / (np.abs(s).sum() + 1e-8) + m

        p = max(p, self.min_rating)
        p = min(p, self.max_rating)

        return p

    def predict(self, u, i):
        return self.__predict(u, i) if self.user_knn else self.__predict(i, u)

    def add(self, x, y, r):
        self.rating[x][y] = r
        self.rated[x][y] = 1.0


