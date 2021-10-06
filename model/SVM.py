# -*- coding = utf-8 -*-
# @Time : 2021/9/27 1:11
# @Author : CDC
# @File : SVM.py
# @Software: PyCharm
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np

class SVM():

    def __init__(self):
        self.classifier = svm.SVC(C=1, kernel='linear', gamma=1, decision_function_shape='ovo') # ovo:一对一策略

    def fit(self, data, label):
        self.classifier.fit(data,label)

    def predict(self, data):
        predictions = self.classifier.predict(data)

        return predictions