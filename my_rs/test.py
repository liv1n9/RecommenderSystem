import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

if __name__ == "__main__":
    a = np.array([[1], [2], [3], [4], [5]])
    b = np.sum(a, axis=0)
    print(b)



