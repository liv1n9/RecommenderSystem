import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class KNN:
    def __init__(self, data, user_knn=True, content_base=True, ld1=4, ld2=0.2, ld3=10, ld4=1, ld5=0.75):
        self.matrix = data.matrix
        self.rated = data.rated

        self.n_users = data.n_users
        self.n_items = data.n_items
        self.min_rating = data.min_rating
        self.max_rating = data.max_rating
        self.user_knn = user_knn
        self.content_base = content_base

        self.att_items = data.att_items
        self.att_users = data.att_users
        self.info_items = data.info_items
        self.info_users = data.info_users

        self.ld1 = ld1
        self.ld2 = ld2
        self.ld3 = ld3
        self.ld4 = ld4
        self.ld5 = ld5 * self.att_items.shape[1] if self.user_knn else ld5 * self.att_users.shape[1]

        self.n = self.n_users if self.user_knn else self.n_items
        self.ld4 = int(self.ld4 * self.n)
        self.m = np.zeros(self.n)
        self.neighbor = [None] * self.n

    def __compute_utility(self):
        for x in range(self.n):
            y = np.where(self.rated[x] == 1)[0] if self.user_knn else np.where(self.rated[:, x] == 1)[
                0]
            r = self.matrix[x, y] if self.user_knn else self.matrix[y, x]
            self.m[x] = np.mean(r) if r.size > 0 else 0.0
        if self.user_knn:
            self.c = np.dot(self.rated, self.rated.T)
        else:
            self.c = np.dot(self.rated.T, self.rated)

        if self.content_base:
            i_x = np.dot(self.matrix.T, self.att_users)
            i_y = np.dot(self.rated.T, self.att_users)
            theta = np.mean(i_y) * 2 / 3
            i_y[i_y < theta] = theta
            self.info_items = i_x / i_y

            u_x = np.dot(self.matrix, self.att_items)
            u_y = np.dot(self.rated, self.att_items)
            theta = np.mean(u_y) * 2 / 3
            u_y[u_y < theta] = theta
            self.info_users = u_x / u_y

            if self.user_knn:
                self.d = np.where(self.info_users > 0, 1, 0)
                self.d = np.dot(self.d, self.d.T)
            else:
                self.d = np.where(self.info_items > 0, 1, 0)
                self.d = np.dot(self.d, self.d.T)

    def __compute_similarity(self):
        if self.content_base:
            if self.user_knn:
                matrix = np.concatenate((self.matrix, self.info_users), axis=1)
            else:
                matrix = np.concatenate((self.matrix, self.info_items.T))
        else:
            matrix = self.matrix
        self.s = cosine_similarity(matrix) if self.user_knn else cosine_similarity(matrix.T)

    def __compute_neighbor(self):
        for x in range(self.n):
            y = np.where(self.c[x] >= self.ld1)[0]
            y = y[y != x]
            if self.content_base:
                d = self.d[x, y]
                ids = np.where(d > self.ld5)[0]
                y = y[ids]

            sim = self.s[x][y]
            ids = np.where(sim >= self.ld2)[0]
            y = y[ids]
            sim = self.s[x][y]
            ids = np.argsort(sim)[-self.ld4:]
            self.neighbor[x] = y[ids]

    def compute(self):
        self.__compute_utility()
        self.__compute_similarity()
        self.__compute_neighbor()

    def add(self, u, i, r):
        self.matrix[u][i] = r
        self.rated[u][i] = 1

    def predict(self, u, i):
        if self.user_knn:
            r = self.matrix[self.neighbor[u], i] - self.m[self.neighbor[u]]
            s = self.s[u][self.neighbor[u]] * self.rated[self.neighbor[u], i]
            result = (r * s).sum() / (np.abs(s).sum() + 1e-8) + self.m[u]
        else:
            r = self.matrix[u, self.neighbor[i]] - self.m[self.neighbor[i]]
            s = self.s[i][self.neighbor[i]] * self.rated[u, self.neighbor[i]]
            result = (r * s).sum() / (np.abs(s).sum() + 1e-8) + self.m[i]
        result = max(result, self.min_rating)
        result = min(result, self.max_rating)

        return result

    def predict_group(self, x):
        if self.neighbor[x].size == 0:
            return np.array([]), np.array([])

        matrix = self.matrix if self.user_knn else self.matrix.T
        rated = self.rated if self.user_knn else self.rated.T
        mask = np.sum(rated[self.neighbor[x]], axis=0)
        mask = mask * (1 - rated[x])
        y = np.where(mask >= self.ld3)[0]

        s = self.s[x][self.neighbor[x]]
        rated_y = rated[self.neighbor[x]][:, y]
        ratings_y = (matrix[self.neighbor[x]][:, y] - np.array([self.m[self.neighbor[x]]]).T) * rated_y
        results = np.dot(s, ratings_y) / (np.dot(np.abs(s), rated_y) + 1e-8) + self.m[x]
        results[results < self.min_rating] = self.min_rating
        results[results > self.max_rating] = self.max_rating

        return y, results
