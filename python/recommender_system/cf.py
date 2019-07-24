import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity


class CF:
    def __init__(self, r_train, k=30, sim_func=cosine_similarity, user_cf=True):
        self.user_cf = user_cf
        if self.user_cf:
            self.r_train = r_train[:, [0, 1, 2]]
        else:
            self.r_train = r_train[:, [1, 0, 2]]
        self.k = k
        self.sim_func = sim_func
        self.n_users = int(np.max(self.r_train[:, 0])) + 1
        self.n_items = int(np.max(self.r_train[:, 1])) + 1
        self.min_rating = np.min(self.r_train[:, 2])
        self.max_rating = np.max(self.r_train[:, 2])
        self.mean_rating = np.mean(self.r_train[:, 2])
        self.r_bar = None
        self.mu = None
        self.mi = None
        self.bu = None
        self.bi = None
        self.s = None
        self.interact_u = None
        self.interact_i = None
        self.ld1 = 15
        self.ld2 = 20
        if not self.user_cf:
            self.ld1, self.ld2 = self.ld2, self.ld1

    def baseline(self, u, i):
        return self.mean_rating + self.bu[u] + self.bi[i]

    def __compute_i(self):
        items = self.r_train[:, 1]
        for i in range(self.n_items):
            ids = np.where(items == i)[0].astype(np.int32)
            self.interact_i[i] = ids.size
            ratings = self.r_train[ids, 2]
            self.mi[i] = np.mean(ratings) if ids.size > 0 else 0
            users = self.r_train[ids, 0].astype(np.int32)
            self.bi[i] = np.sum(ratings - self.bu[users] - self.mean_rating) / (self.ld1 + ids.size)

    def __compute_u(self):
        users = self.r_train[:, 0]
        for u in range(self.n_users):
            ids = np.where(users == u)[0].astype(np.int32)
            self.interact_u[u] = ids.size
            ratings = self.r_train[ids, 2]
            self.mu[u] = np.mean(ratings) if ids.size > 0 else 0
            items = self.r_train[ids, 1].astype(np.int32)
            self.bu[u] = np.sum(ratings - self.bi[items] - self.mean_rating) / (self.ld2 + ids.size)

    def fit(self):
        self.r_bar = self.r_train.copy()
        self.mu = np.zeros(self.n_users)
        self.mi = np.zeros(self.n_items)
        self.bu = np.zeros(self.n_users)
        self.bi = np.zeros(self.n_items)
        self.interact_u = np.zeros(self.n_users)
        self.interact_i = np.zeros(self.n_items)
        for i in range(5):
            if self.user_cf:
                self.__compute_i()
                self.__compute_u()
            else:
                self.__compute_u()
                self.__compute_i()

        self.r_bar = coo_matrix((self.r_bar[:, 2], (self.r_bar[:, 0], self.r_bar[:, 1])),
                                (self.n_users, self.n_items))
        self.s = self.sim_func(self.r_bar.toarray())
        for x in range(self.r_bar.data.size):
            u = int(self.r_bar.row[x])
            i = int(self.r_bar.col[x])
            self.r_bar.data[x] -= self.baseline(u, i)
        self.r_bar = self.r_bar.toarray()

    def __predict(self, u, i):
        result = 0.0
        if u >= self.n_users:
            if i >= self.n_items:
                result = self.mean_rating
            else:
                result = self.bi[i] + self.mean_rating
        elif i >= self.n_items:
            result = self.bu[u] + self.mean_rating
        else:
            ids = np.where(self.r_train[:, 1] == i)[0].astype(np.int32)
            users_rated_i = (self.r_train[ids, 0]).astype(np.int32)
            sim = self.s[u, users_rated_i]
            nns = np.argsort(sim)[-self.k:]
            nearest_s = sim[nns]
            r = self.r_bar[users_rated_i[nns], i]
            result = (r * nearest_s).sum() / (nearest_s.sum() + 1e-8) + self.baseline(u, i)
        if result < self.min_rating:
            result = self.min_rating
        if result > self.max_rating:
            result = self.max_rating
        return result

    def predict(self, u, i):
        return self.__predict(u, i) if self.user_cf else self.__predict(i, u)
