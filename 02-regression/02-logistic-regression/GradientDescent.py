# Linear Regression : It is only for study
# Copyright : yoonkt200@gmail.com
# Apache License 2.0
# =============================================================================

import numpy as np
from sklearn.metrics import roc_auc_score


class LogisticRegression():
    def __init__(self, lr, epoch, train_data, valid_data):
        self._lr = lr
        self._epoch = epoch
        self._train_file_path = train_data
        self._valid_file_path = valid_data
        self._valid_loss_list = []

    def _load_dataset(self):
        train_file = open(self._train_file_path, 'r')
        valid_file = open(self._valid_file_path, 'r')
        self._train_data = train_file.read().split('\n')
        self._valid_data = valid_file.read().split('\n')
        train_file.close()
        valid_file.close()

        # find max index
        self.feature_max_index = 0
        print("Start to init FM vectors.")
        for row in self._train_data:
            for element in row.split(" ")[1:]:
                index = int(element.split(":")[0])
                if self.feature_max_index < index:
                    self.feature_max_index = index

        for row in self._valid_data:
            for element in row.split(" ")[1:]:
                index = int(element.split(":")[0])
                if self.feature_max_index < index:
                    self.feature_max_index = index

        # init FM vectors
        self._init_vectors()
        print("Finish init FM vectors.")

    def _init_vectors(self):
        self.w = np.random.randn(self.feature_max_index+1)
        self.train_x_data = np.zeros((len(self._train_data), self.feature_max_index+1))
        self.train_y_data = np.zeros((len(self._train_data)))
        self.valid_x_data = np.zeros((len(self._valid_data), self.feature_max_index+1))
        self.valid_y_data = np.zeros((len(self._valid_data)))

        # make numpy dataset
        for n, row in enumerate(self._train_data):
            element = row.split(" ")
            self.train_y_data[n] = int(element[0])
            self.train_x_data[n][np.array([int(pair.split(":")[0]) for pair in element[1:]])] = np.array(
                [int(pair.split(":")[1]) for pair in element[1:]])

        for n, row in enumerate(self._valid_data):
            element = row.split(" ")
            self.valid_y_data[n] = int(element[0])
            self.valid_x_data[n][np.array([int(pair.split(":")[0]) for pair in element[1:]])] = np.array(
                [int(pair.split(":")[1]) for pair in element[1:]])

    def train(self):
        self._load_dataset()
        for epoch in range(self._epoch):
            y_hat = self.sigmoid(self.train_x_data)
            valid_y_hat = self.sigmoid(self.valid_x_data)
            loss = self.log_likelihood(self.train_y_data, y_hat)
            valid_loss = self.log_likelihood(self.valid_y_data, valid_y_hat)
            roc_auc = roc_auc_score(self.train_y_data, y_hat)
            valid_roc_auc = roc_auc_score(self.valid_y_data, valid_y_hat)
            self._print_learning_info(epoch=epoch, train_loss=loss, valid_loss=valid_loss,
                                      train_auc=roc_auc, valid_auc=valid_roc_auc)
            gradient = self.gradient(self.train_x_data, self.train_y_data, y_hat)
            self.w = self.w - self._lr * gradient

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.dot(x, self.w)))

    def log_likelihood(self, y, y_hat):
        return -(np.dot(y, np.log(y_hat)) + np.dot(1-y, np.log(1-y_hat)))

    def gradient(self, x, y, y_hat):
        n = len(y)
        gradient = np.dot(x.transpose(), y_hat-y) / n
        return gradient

    def _print_learning_info(self, epoch, train_loss, valid_loss, train_auc, valid_auc):
        print("epoch:", epoch, "||", "train_loss:", train_loss, "||", "valid_loss:", valid_loss,
              "||", "Train AUC:", train_auc, "||", "Test AUC:", valid_auc)

    def print_results(self):
        print(self.w)


if __name__ == "__main__":
    lr = LogisticRegression(lr=0.05,
                            epoch=1000,
                            train_data='../../dataset/train.txt',
                            valid_data='../../dataset/test.txt')
    lr.train()
    lr.print_results()