from recommender_system.knn.knn import KNN


class KNNCoTraining:
    def __init__(self, data, user_first=True, content_base=False, loop=10, ld1=4, ld2=0.8, ld3=10, ld4=1, ld5=0.75):
        self.data = data
        self.user_first = user_first
        self.content_base = content_base
        self.loop = loop

        self.ld1 = ld1
        self.ld2 = ld2
        self.ld3 = ld3
        self.ld4 = ld4
        self.ld5 = ld5

        self.knn_model = [None] * 2
        self.t = 0
        self.total = 0

    def compute(self):
        for j in range(2):
            self.knn_model[j] = KNN(
                data=self.data,
                user_knn=bool(self.user_first ^ j),
                content_base=self.content_base,
                ld1=self.ld1,
                ld2=self.ld2,
                ld3=self.ld3,
                ld4=self.ld4,
                ld5=self.ld5
            )
        enrich = -1
        while enrich != 0 and self.t < self.loop:
            enrich = 0
            print('loop:', self.t)
            self.t += 1
            for j in range(2):
                self.knn_model[j].compute()
                temp_y = [None] * self.knn_model[j].n
                temp_r = [None] * self.knn_model[j].n
                for x in range(self.knn_model[j].n):
                    temp_y[x], temp_r[x] = self.knn_model[j].predict_group(x)
                    enrich += temp_y[x].size
                for x in range(self.knn_model[j].n):
                    for i in range(temp_y[x].size):
                        if self.knn_model[j].user_knn:
                            self.knn_model[j ^ 1].add(x, temp_y[x][i], temp_r[x][i])
                        else:
                            self.knn_model[j ^ 1].add(temp_y[x][i], x, temp_r[x][i])
            self.total += enrich
            print('enrich:', enrich)

        for j in range(2):
            self.knn_model[j].compute()

    def predict(self, u, i):
        if self.data.rated[u][i] == 1:
            return self.data.matrix[u][i]
        return (self.knn_model[0].predict(u, i) + self.knn_model[1].predict(u, i)) / 2
