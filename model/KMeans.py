# -*- coding = utf-8 -*-
# @Time : 2021/9/21 15:31
# @Author : CDC
# @File : demo1.py
# @Software: PyCharm
import random
import numpy as np


class RBF_KMeans():

    def __init__(self, center_num, data_train, label_train):

        self.center_num = center_num
        self.center = None
        self.data_train = data_train
        self.label_train = label_train
        self.sigma = None
        self.omega = None

    def center_random_select(self):
        center = np.zeros((self.center_num, len(self.data_train[0])))
        for i in range(self.center_num):
            sample = random.choices(self.data_train)
            center[i] = sample[0]
        return center

    def set_center(self, center):
        self.center = center

    def Kmeans_sigma(self):
        self.sigma = 1

    def rbf_kernel(self, data, center):
        return np.exp(-np.linalg.norm(data - center) ** 2 / (2 * self.sigma ** 2))

    def matrix_calculation(self, data_used):

        fai = np.zeros((len(data_used), self.center_num))

        for i in range(len(data_used)):
            for j in range(self.center_num):
                fai[i, j] = self.rbf_kernel(data_used[i], self.center[j])
        return fai

    def fit(self, data_used):
        fai = self.matrix_calculation(data_used)
        self.omega = np.dot(np.linalg.pinv(fai), self.label_train)

    def prediction(self, data_used):
        fai = self.matrix_calculation(data_used)
        prediction = np.dot(fai, self.omega)
        return prediction
