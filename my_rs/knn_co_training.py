from my_rs.knn import KNN


class KNNCoTraining:
    def __init__(self, data, loop=10, m=10, k=30):
        self.data = data
        self.loop = loop
        self.m = m
        self.k = k
        self.knn_model = [None] * 2

    def compute(self):
        for j in range(2):
            self.knn_model[j] = KNN(self.data, bool(j), self.k)

        loop = 0
        while loop < self.loop:
            print('loop', loop)
            enrich_y = [None] * 2
            enrich_r = [None] * 2
            for j in range(2):
                self.knn_model[j].compute()
                temp_y = [None] * self.knn_model[j].n_x
                temp_r = [None] * self.knn_model[j].n_x
                for x in range(self.knn_model[j].n_x):
                    temp_y[x], temp_r[x] = self.knn_model[j].predict_group(x, self.m)
                enrich_y[j] = temp_y
                enrich_r[j] = temp_r

            for j in range(2):
                for x in range(self.knn_model[j].n_x):
                    for i in range(enrich_r[j][x].size):
                        if self.knn_model[j].user_knn:
                            self.knn_model[j ^ 1].add(x, enrich_y[j][x][i], enrich_r[j][x][i])
                        else:
                            self.knn_model[j ^ 1].add(enrich_y[j][x][i], x, enrich_r[j][x][i])
            loop += 1

        for j in range(2):
            self.knn_model[j].compute()

    def predict(self, u, i):
        result = (self.knn_model[0].predict(u, i) + self.knn_model[1].predict(u, i)) / 2
        result = max(result, self.data.min_rating)
        result = min(result, self.data.max_rating)
        return result


