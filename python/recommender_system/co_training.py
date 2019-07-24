import numpy as np
from cf import CF


class CoTraining:
    def __init__(self, r_train, m=200, x=30, loop=20, user_first=False):
        self.r_train = r_train[:, [0, 1, 2]]
        self.m = m
        self.x = x
        self.loop = loop
        self.user_first = user_first
        self.n_users = int(np.max(self.r_train[:, 0])) + 1
        self.n_items = int(np.max(self.r_train[:, 1])) + 1

        self.cf = [None] * 2

        self.t_l = [None] * 2
        self.t_ul = np.zeros((0, 2))

    def __conf(self, r, u, i, j):
        return (np.sqrt(self.cf[j].interact_u[u]) * self.cf[j].interact_i[i]) / (abs(r - self.cf[j].baseline(u, i)) + 1e-8)

    def conf(self, r, u, i, j):
        return self.__conf(r, u, i, j) if self.cf[j].user_cf else self.__conf(r, i, u, j)

    def fit(self):
        users = self.r_train[:, 0]
        items = np.arange(self.n_items)
        for u in range(self.n_users):
            ids = np.where(users == u)[0].astype(np.int32)
            rated_items = self.r_train[ids, 1]
            mask = np.isin(items, rated_items)
            unrated_items = np.where(mask == False)[0].astype(np.int32)
            pick = max(0, int(self.x - rated_items.shape[0]))
            random_pick = np.random.permutation(unrated_items.shape[0])[:pick]
            temp = np.zeros((pick, 2))
            for x, i in np.ndenumerate(unrated_items[random_pick]):
                temp[x[0]][0] = u
                temp[x[0]][1] = i
            self.t_ul = np.concatenate((self.t_ul, temp))

        self.m = min(self.m, int(self.t_ul.shape[0] / 20))

        self.t_l[0] = self.r_train.copy()
        self.t_l[1] = self.r_train.copy()
        for it in range(self.loop):
            print('loop', it)
            if self.t_ul.shape[0] == 0:
                break
            for j in range(2):
                self.cf[j] = CF(r_train=self.t_l[j], user_cf=bool(self.user_first ^ j))
                self.cf[j].fit()

                r_list = np.zeros((self.t_ul.shape[0], 1))
                conf_list = np.zeros(self.t_ul.shape[0])

                for x in range(self.t_ul.shape[0]):
                    u = int(self.t_ul[x][0])
                    i = int(self.t_ul[x][1])
                    r_list[x][0] = self.cf[j].predict(u, i)
                    conf_list[x] = self.conf(r_list[x], u, i, j)

                most_conf = np.argsort(conf_list)[-self.m:]
                temp = np.concatenate((self.t_ul, r_list), axis=1)
                self.t_l[j ^ 1] = np.concatenate((self.t_l[j ^ 1], temp[most_conf]))

                index = np.arange(self.t_ul.shape[0])
                mask = np.isin(index, most_conf)
                ids = np.where(mask == False)[0].astype(np.int32)
                self.t_ul = self.t_ul[ids]

        for j in range(2):
            self.cf[j] = CF(r_train=self.t_l[j], user_cf=bool(self.user_first ^ j))
            self.cf[j].fit()

    def predict(self, u, i):
        h1 = self.cf[0].predict(u, i)
        h2 = self.cf[1].predict(u, i)
        return (h1 + h2) / 2
