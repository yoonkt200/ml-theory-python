import numpy as np


class ALS:
    def __init__(self, R, k, r_lambda, confidence, epochs, verbose=False):
        """
        :param R: rating matrix
        :param k: latent parameter
        :param r_lambda: alpha on ALS
        :param confidence: confidence on ALS
        :param epochs: training epochs
        :param verbose: print status
        """
        self._R = R
        self._num_users, self._num_items = R.shape
        self._k = k
        self._r_lambda = r_lambda
        self._confidence = confidence
        self._epochs = epochs
        self._verbose = verbose
        self._init_vectors()

    def _init_vectors(self):
        # init latent features
        self._U = np.random.normal(size=(self._num_users, self._k))
        self._I = np.random.normal(size=(self._num_items, self._k))

        # make binary rating matrix P
        self._P = np.copy(self._R)
        self._P[self._P > 0] = 1

        # make confidence matrix C
        self._C = 1 + self._confidence * self._R

    def fit(self):
        """
        fit ALS model
        """
        # train while epochs
        self._training_process = []
        for epoch in range(self._epochs):
            self.alternating_least_squares()
            cost = self.cost()
            self._training_process.append((epoch, cost))

            # print status
            if self._verbose == True and ((epoch + 1) % 10 == 0):
                print("Iteration: %d ; cost = %.4f" % (epoch + 1, cost))

    def cost(self):
        """
        compute root mean square error
        :return: rmse cost
        """
        r_hat = self.predict()
        confidence_loss = np.sum(np.power(self._C * (self._P - r_hat), 2))
        regularization_term = self._r_lambda * (np.sum(np.power(self._U, 2)) + np.sum(np.power(self._I, 2)))
        return np.sqrt(confidence_loss + regularization_term)

    def alternating_least_squares(self):
        """
        ALS optimizer
        """
        # update_y_u
        for u in range(self._U.shape[0]):
            C_u = np.diag(self._C[u])
            left_equation = self._I.T.dot(C_u).dot(self._I) + np.dot(self._r_lambda, np.identity(self._U.shape[1]))
            right_equation = self._I.T.dot(C_u).dot(self._P[u])
            self._U[u] = np.linalg.inv(left_equation).dot(right_equation)

        # update x_i
        for i in range(self._I.shape[0]):
            C_i = np.diag(self._C[:, i])
            left_equation = self._U.T.dot(C_i).dot(self._U) + np.dot(self._r_lambda, np.identity(self._I.shape[1]))
            right_equation = self._U.T.dot(C_i).dot(self._P[:, i])
            self._I[i] = np.linalg.inv(left_equation).dot(right_equation)

    def predict(self):
        """
        calc complete matrix U X I
        """
        return self._U.dot(self._I.T)

    def print_results(self):
        """
        print fit results
        """
        print("Final RMSE:")
        print(self._training_process[self._epochs-1][1])
        print("Final R matrix:")
        for row in self.predict():
            print(["{0:0.2f}".format(i) for i in row])


# run example
if __name__ == "__main__":
    # rating matrix - User X Item : (7 X 5)
    R = np.array([
        [1, 0, 0, 0, 3],
        [2, 0, 0, 1, 0],
        [0, 2, 0, 5, 0],
        [1, 0, 0, 0, 4],
        [0, 0, 5, 4, 0],
        [0, 1, 0, 4, 0],
        [0, 0, 0, 1, 0],
    ])

    # P, Q is (7 X k), (k X 5) matrix
    factorizer = ALS(R, k=8, r_lambda=40, confidence=40, epochs=50, verbose=True)
    factorizer.fit()
    factorizer.print_results()
