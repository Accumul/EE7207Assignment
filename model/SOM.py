# -*- coding = utf-8 -*-
# @Time : 2021/9/21 15:31
# @Author : CDC
# @File : demo1.py
# @Software: PyCharm
import random
import numpy as np


class SOM():
    def __init__(self, center_num_x, center_num_y, data_train, label_train):
        self.center_num_x = center_num_x
        self.center_num_y = center_num_y
        self.center_num = center_num_x * center_num_y
        self.data_train = data_train
        self.label_train = label_train
        self.sigma = 1
        self.data_len = len(data_train[0])
        self.center = np.zeros((self.center_num, self.data_len))
        self.sample = None
        self.omega = None

    def Initialization(self):
        for i in range(self.center_num):
            for j in range(self.data_len):
                self.center[i][j] = random.uniform(-1, 1)

    def sampling(self):
        return random.choices(self.data_train)

    def competition(self):

        self.sample = self.sampling()
        min_dist = 100

        for i in range(self.center_num):
            center_dist = np.linalg.norm(self.center[i] - self.sample)
            if center_dist < min_dist:
                min_dist = center_dist
                win_index = i

        return win_index

    def FindNeuronPlace(self, i):
        win_y = int(i / self.center_num_x) + 1
        win_x = i % self.center_num_x + 1
        return win_x, win_y

    def sigma0(self):
        return (((self.center_num_x - 1) * (self.center_num_y - 1)) ** 0.5) / 2

    def iteration(self):
        sigma0 = self.sigma0()
        tao1 = 1000 / (np.log(sigma0))
        for n in range(1000):
            win_index = self.competition()
            win_x, win_y = self.FindNeuronPlace(win_index)
            learn_rate = 0.1 * np.exp(-n / 1000)
            self.sigma = sigma0 * np.exp(-n / tao1)
            for i in range(self.center_num):
                x, y = self.FindNeuronPlace(i)
                hji = np.exp(-((win_x - x) ** 2 + (win_y - y) ** 2) / (2 * self.sigma ** 2))
                self.center[i] = self.center[i] + learn_rate * hji * (self.sample - self.center[i])

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
