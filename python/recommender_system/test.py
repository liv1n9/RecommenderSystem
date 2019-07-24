import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix
import itertools

if __name__ == '__main__':
    row = np.array([0, 3, 1, 0])
    col = np.array([0, 3, 1, 2])
    data = np.array([4, 5, 7, 9])
    a = coo_matrix((data, (row, col)), shape=(4, 4))
    print(a.toarray())
    print(a)
