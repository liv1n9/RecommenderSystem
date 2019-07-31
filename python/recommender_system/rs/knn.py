import numpy as np


class KNN:
    def __init__(self, data, user_knn=True, ld1=3, ld2=5, k=200):
        # Ma trận xếp hạng
        self.matrix = data.matrix

        # Ma trận trạng thái đánh giá. self.rated[u][i] == True tương đương (u,i) đã được xếp hạng
        self.rated = data.rated

        self.n_users = data.n_users
        self.n_items = data.n_items
        self.min_rating = data.min_rating
        self.max_rating = data.max_rating
        self.user_knn = user_knn

        # lambda_1 và lambda_2
        self.ld1 = ld1
        self.ld2 = ld2

        # Số láng giềng gần nhất
        self.k = k

        self.n = self.n_users if self.user_knn else self.n_items

        # Trung bình xếp hạng
        self.m = np.zeros(self.n)

        # c[x][y] là số sản phẩm x, y cùng xếp hạng, hoặc số người dùng cùng đánh giá x, y, tuỳ theo model
        self.c = None

        # s[x][y] là độ tương tự
        self.s = None

        self.neighbor = [None] * self.n

    def __compute_mean(self):
        for x in range(self.n):
            y = np.where(self.rated[x] == 1)[0].astype(np.int32) if self.user_knn else np.where(self.rated[:, x] == 1)[0].astype(np.int32)
            r = self.matrix[x, y] if self.user_knn else self.matrix[y, x]
            self.m[x] = np.mean(r) if r.size > 0 else 0.0

    # Tính Pearson theo bằng vectorized
    def __compute_similarity(self):
        if self.user_knn:
            diff = (self.matrix - np.array([self.m]).T) * self.rated
            self.c = np.dot(self.rated, self.rated.T)
            x = np.dot(diff, diff.T)
            y = np.sqrt(np.dot(diff ** 2, self.rated.T))
            y = y * y.T
            z = self.c / (self.c + 100)
            self.s = x / (y + 1e-8) * z
        else:
            diff = (self.matrix.T - np.array([self.m]).T) * self.rated.T
            self.c = np.dot(self.rated.T, self.rated)
            x = np.dot(diff, diff.T)
            y = np.sqrt(np.dot(diff ** 2, self.rated))
            y = y * y.T
            z = self.c / (self.c + 100)
            self.s = x / (y + 1e-8) * z

    def __compute_neighbor(self):
        for x in range(self.n):
            # Chọn ra láng giềng y thoả mãn c[x][y] lớn hơn lambda_1 (tập sinh)
            y = np.where(self.c[x] >= self.ld1)[0].astype(np.int32)

            # Bỏ x ra khỏi láng giềng của chính x
            y = y[y != x]

            # Lấy ra độ tương tự của x so với các láng giềng y
            sim = self.s[x][y]

            # Sắp xếp tăng dần và chọn ra k láng giềng cuối cùng
            ids = np.argsort(sim)[-self.k:]

            self.neighbor[x] = y[ids]

    def compute(self):
        self.__compute_mean()
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

    # Tính xếp hạng của các cặp (x, y) chưa có nhãn bằng vectorized
    def predict_group(self, x):
        if self.neighbor[x].size == 0:
            return np.array([]), np.array([])

        # Xoay các ma trận tuỳ theo model
        matrix = self.matrix if self.user_knn else self.matrix.T
        rated = self.rated if self.user_knn else self.rated.T

        mask = np.sum(rated[self.neighbor[x]], axis=0)
        mask = mask * (1 - rated[x])

        # Chọn ra các y được nhiều hơn lambda_2 láng giềng của x tác động tới
        y = np.where(mask >= self.ld2)[0].astype(np.int32)

        # Tính bằng vectorized, kết quả dự đoán của các y lưu trong results
        s = self.s[x][self.neighbor[x]]
        rated_y = rated[self.neighbor[x]][:, y]
        ratings_y = (matrix[self.neighbor[x]][:, y] - np.array([self.m[self.neighbor[x]]]).T) * rated_y
        results = np.dot(s, ratings_y) / (np.dot(np.abs(s), rated_y) + 1e-8) + self.m[x]
        results[results < self.min_rating] = self.min_rating
        results[results > self.max_rating] = self.max_rating
        return y, results
