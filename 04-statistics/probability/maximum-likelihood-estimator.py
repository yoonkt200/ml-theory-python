import numpy as np
import math


class MLE():
    def __init__(self, samples, m, std, learning_rate, epochs, verbose=False):
        """
        :param samples: samples for get MLE
        :param learning_rate: alpha on weight update
        :param epochs: training epochs
        :param verbose: print status
        """
        self._samples = samples
        self._m = m
        self._std = std
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._verbose = verbose


    def pdf(self, x):
        """
        Probability Density Function
        :param x:
        :return: likelihood of input x (same as y of pdf)
        """
        return (1 / math.sqrt(2*math.pi) * math.pow(self._std, 2)) * math.exp(-(pow(x-self._m, 2) / (2*math.pow(self._std, 2))))


    def fit(self):
        pass


    def cost(self):
        pass


    def update(self):
        pass