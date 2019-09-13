from distutils.core import setup
from Cython.Build import cythonize

if __name__ == "__main__":
    setup(ext_modules=cythonize("./recommender_system/knn/knn.pyx", annotate=False, language_level=3))
