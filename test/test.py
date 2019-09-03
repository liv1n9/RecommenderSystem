import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class A:
    def __init__(self, data):
        self.data = data

    def inc(self):
        self.data[0] += 1


class B:
    def __init__(self, data):
        self.data = data


if __name__ == '__main__':
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 2, 3]])
    b = np.array([1, 3, 4])
    c = a - np.array([b]).T
    print(c)
