import numpy as np


class A:
    def __init__(self, data):
        self.data = data

    def inc(self):
        self.data[0] += 1


class B:
    def __init__(self, data):
        self.data = data


if __name__ == '__main__':
    a = np.array([[1, 2, 3, 4]])
    b = np.repeat(a, 5, axis=0).T
    print(b)
