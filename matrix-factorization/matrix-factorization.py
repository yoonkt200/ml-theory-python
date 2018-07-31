import numpy as np


class MatrixFactorization():

    def __init__(self, R, k, learning_rate, reg_param, epochs):
        self._R = R
        self._k = k
        self._learning_rate = learning_rate
        self._reg_param = reg_param
        self._epochs = epochs

    def fit(self):
        pass

    def error(self):
        pass

    def sgd(self):
        pass

    def get_prediction(self):
        pass

    def get_R(self):
        pass

    def print_results(self):
        pass


# run example
if __name__ == "__main__":
    R = np.array([
        [1, 0, 0, 1, 3],
        [2, 0, 3, 1, 1],
        [1, 2, 0, 5, 0],
        [1, 0, 0, 4, 4],
        [2, 1, 5, 4, 0],
        [5, 1, 5, 4, 0],
    ])

    factorizer = MatrixFactorization(R, k=3, learning_rate=0.01, reg_param=0.01, epochs=100)
    factorizer.fit()
    factorizer.print_results()