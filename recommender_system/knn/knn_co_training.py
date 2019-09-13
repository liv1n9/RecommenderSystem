from recommender_system.knn.knn import KNN


class KNNCoTraining:
    def __init__(self, data, user_knn, a0=14, c0=0.5, c1=1, d0=10, content_base=False, a1=0.75, b0=0.5, b1=0.8, loop=10):
        self.data = data
        self.user_knn = user_knn
        self.content_base = content_base
        self.loop = loop

        self.a0 = a0
        self.c0 = c0
        self.c1 = c1
        self.d0 = d0
        self.content_base = content_base
        self.a1 = a1
        self.b0 = b0
        self.b1 = b1

        self.knn_model = [None] * 2
        self.t = 0
        self.total = 0

    def compute(self):
        for j in range(2):
            self.knn_model[j] = KNN(
                data=self.data,
                user_knn=bool(self.user_knn ^ j),
                a0=self.a0,
                c0=self.c0,
                c1=self.c1,
                d0=self.d0,
                content_base=self.content_base,
                a1=self.a1,
                b0=self.b0,
                b1=self.b1
            )
        enrich = -1
        while enrich != 0 and self.t < self.loop:
            enrich = 0
            print('loop:', self.t)
            self.t += 1
            for j in range(2):
                self.knn_model[j].compute()
                temp = list(map(self.knn_model[j].predict_group, range(self.knn_model[j].n)))
                enrich += sum(t[0].size for t in temp)
                for x in range(self.knn_model[j].n):
                    y, r = temp[x]
                    for i in range(y.size):
                        self.knn_model[j ^ 1].add(y[i], x, r[i])
                self.knn_model[j].c1 = int(self.knn_model[j].c1 * 0.6)
            self.total += enrich
            print('enrich:', enrich)

        for j in range(2):
            self.knn_model[j].compute()

    def predict(self, u, i):
        if self.data.rated[u][i] == 1:
            return self.data.rating[u][i]
        return (self.knn_model[0].predict(u, i) + self.knn_model[1].predict(u, i)) / 2
