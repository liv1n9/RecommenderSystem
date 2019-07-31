from rs.knn import KNN


class KNNCoTraining:
    def __init__(self, data, user_first=True, loop=15, model=KNN, ld1=3, ld2=5, k=50):
        self.data = data
        self.user_first = user_first
        self.loop = loop

        # Model User/Item-based. Em cài thêm model KNNBaseline (dùng ước tính cơ sở) nhưng không thấy khả quan hơn
        self.model = model

        self.ld1 = ld1
        self.ld2 = ld2
        self.k = k

        self.knn_model = [None] * 2

    def compute(self):

        # Khởi tạo models
        for j in range(2):
            self.knn_model[j] = self.model(self.data, bool(self.user_first ^ j), self.ld1, self.ld2, self.k)

        # enrich thể hiện số dữ liệu được gán nhãn thêm trong mỗi vòng lặp
        enrich = -1

        loop = 0
        while enrich != 0 and loop < self.loop:
            enrich = 0
            loop += 1
            print('loop', loop)
            for j in range(2):
                self.knn_model[j].compute()
                temp_y = [None] * self.knn_model[j].n
                temp_r = [None] * self.knn_model[j].n
                for x in range(self.knn_model[j].n):
                    # Với mỗi x, dùng hàm predict_group để dự đoán các y chắc chắn
                    temp_y[x], temp_r[x] = self.knn_model[j].predict_group(x)
                    enrich += temp_y[x].size

                # Sau khi dự đoán xong kết nạp vào ma trận xếp hạng
                for x in range(self.knn_model[j].n):
                    for i in range(temp_y[x].size):
                        if self.knn_model[j].user_knn:
                            self.knn_model[j ^ 1].add(x, temp_y[x][i], temp_r[x][i])
                        else:
                            self.knn_model[j ^ 1].add(temp_y[x][i], x, temp_r[x][i])

    def predict(self, u, i):
        return self.data.matrix[u][i]






