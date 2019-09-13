import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

if __name__ == "__main__":
    a = np.zeros((3, 4))
    b = a.T
    b[3][2] = 1
    print(a)

