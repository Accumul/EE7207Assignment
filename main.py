# -*- coding = utf-8 -*-
# @Time : 2021/9/26 15:56
# @Author : CDC
# @File : K-MEANS.py
# @Software: PyCharm
import scipy.io as scio
from scipy.io import loadmat
from sklearn.cluster import KMeans
import numpy as np
from model.SVM import SVM
from model.RBF import RBF
from model.SOM import SOM
from model.KMeans import RBF_KMeans
import matplotlib.pyplot as plt

verify_size = 50


def import_data():
    data_train = loadmat('data/data_train')['data_train']  # len = 330
    data_test = loadmat('data/data_test')['data_test']  # len = 21
    label_train = loadmat('data/label_train')['label_train']  # len = 330
    return data_train, label_train.ravel(), data_test


def data_partition(verify_size):
    data_verify = []
    label_verify = []
    data_train, label_train, data_test = import_data()
    for i in range(verify_size):
        n = np.random.randint(0, len(data_train))
        data_verify.append(data_train[n])
        label_verify.append(label_train[n])
        data_train = np.delete(data_train, n, 0)
        label_train = np.delete(label_train, n, 0)
    data_verify = np.array(data_verify)
    label_verify = np.array(label_verify)
    return data_train, label_train, data_verify, label_verify, data_test


def svm_classifier():
    data_train, label_train, data_verify, label_verify, data_test = data_partition(verify_size)

    cls = SVM()
    cls.fit(data_train, label_train)
    prediction = cls.predict(data_verify)

    error_num = np.linalg.norm(label_verify - prediction) ** 2 / 4
    rate = (len(label_verify) - error_num) / len(label_verify)
    mse = np.linalg.norm(prediction - label_verify) ** 2 / len(label_verify)
    SVM_Mse_result.append(mse)
    SVM_Acc_result.append(rate)
    print("SVM的MSE为：", mse)
    print("SVM的精度为：", rate)
    return cls.predict(data_test)


list1 = list(range(1, 30 + 1))
SVM_Mse_result = []
RBF_Mse_result = []
K_Mse_result = []
SOM_Mse_result = []
SVM_Acc_result = []
RBF_Acc_result = []
K_Acc_result = []
SOM_Acc_result = []
K_Acc_train = []
list2 = []
list3 = []
list_nnum = list(range(1, 76))
list_nnum_over = list(range(1, 201))
def rbf_random(center_num_x, center_num_y):
    data_train, label_train, data_verify, label_verify, data_test = data_partition(verify_size)

    rbf_model = RBF(center_num_x, center_num_y, data_train, label_train)
    rbf_model.random_sigma()
    rbf_model.fit(data_train)
    prediction = rbf_model.prediction(data_verify)
    error_num = np.linalg.norm(label_verify - np.sign(prediction)) ** 2 / 4
    rate = (len(label_verify) - error_num) / len(label_verify)
    mse = np.linalg.norm(prediction - label_verify) ** 2 / len(label_verify)
    RBF_Mse_result.append(mse)
    RBF_Acc_result.append(rate)
    print("RBF的MSE为：", mse)
    print("RBF的精度为：", rate)
    return np.sign(rbf_model.prediction(data_test))


def rbf_kmeans(center_num):
    data_train, label_train, data_verify, label_verify, data_test = data_partition(verify_size)

    rbf_model = RBF_KMeans(center_num, data_train, label_train)
    rbf_model.Kmeans_sigma()
    clf = KMeans(n_clusters=center_num)
    clf.fit(data_train)  # 分组
    rbf_model.set_center(clf.cluster_centers_)  # 两组数据点的中心点
    rbf_model.fit(data_train)
    prediction = rbf_model.prediction(data_verify)
    predictionTrain = rbf_model.prediction(data_train)
    error_num = np.linalg.norm(label_verify - np.sign(prediction)) ** 2 / 4
    error_num2 = np.linalg.norm(label_train - np.sign(predictionTrain)) ** 2 / 4

    rate = (len(label_verify) - error_num) / len(label_verify)
    rate2 = (len(label_train) - error_num2) / len(label_train)
    mse = np.linalg.norm(prediction - label_verify) ** 2 / len(label_verify)
    K_Mse_result.append(mse)
    K_Acc_result.append(rate)
    K_Acc_train.append(rate2)
    print("RBF_Kmeans的MSE为：", mse)
    print("RBF_Kmeans的精度为：", rate)
    return np.sign(rbf_model.prediction(data_test))


def rbf_som(center_num_x, center_num_y):
    data_train, label_train, data_verify, label_verify, data_test = data_partition(verify_size)

    rbf_model = SOM(center_num_x, center_num_y, data_train, label_train)
    rbf_model.Initialization()
    rbf_model.iteration()
    rbf_model.fit(data_train)
    prediction = rbf_model.prediction(data_verify)

    error_num = np.linalg.norm(label_verify - np.sign(prediction)) ** 2 / 4
    rate = (len(label_verify) - error_num) / len(label_verify)
    mse = np.linalg.norm(prediction - label_verify) ** 2 / len(label_verify)
    SOM_Mse_result.append(mse)
    SOM_Acc_result.append(rate)
    print("RBF_som的MSE为：", mse)
    print("RBF_som的精度为：", rate)
    return np.sign(rbf_model.prediction(data_test))


def plot_mse():
    for i in range(30):
        svm_classifier()
        rbf_random(8,8)
        rbf_kmeans(64)
        rbf_som(8,8)

    plt.scatter(list1, SVM_Mse_result, c='#FF0000', s=10, label='SVM', marker='v')
    plt.scatter(list1, RBF_Mse_result, c='#7CFC00', s=10, label='RBF-Random', marker='o')
    plt.scatter(list1, SOM_Mse_result, c='#0000FF', s=10, label='RBF-SOM', marker='x')
    plt.scatter(list1, K_Mse_result, c='#FFA500', s=10, label='RBF-KMeans', marker='D')
    plt.title("Performance-MSE", fontsize=22)
    plt.xlabel("Times", fontsize=12)
    plt.ylabel("MSE", fontsize=22)
    plt.axis([0, 31, 0, 2])
    plt.legend()
    plt.savefig(r'F:\Master\NTU\7207-Neural and Fuzzy Systems\K-MSE.png', dpi=300)
    plt.show()


def plot_acc():
    for i in range(30):
        svm_classifier()
        rbf_random(8,8)
        rbf_kmeans(64)
        rbf_som(8,8)
    plt.scatter(list1, SVM_Acc_result, c='#FF0000', s=10, label='SVM', marker='v')
    plt.scatter(list1, RBF_Acc_result, c='#7CFC00', s=10, label='RBF-Random', marker='o')
    plt.scatter(list1, SOM_Acc_result, c='#0000FF', s=10, label='RBF-SOM', marker='x')
    plt.scatter(list1, K_Acc_result, c='#FFA500', s=10, label='RBF-KMeans', marker='D')
    plt.title("Performance-Acc", fontsize=22)
    plt.xlabel("Times", fontsize=12)
    plt.ylabel("Acc", fontsize=22)
    plt.axis([0, 31, 0.5, 1])
    plt.legend()
    plt.savefig(r'C:\Users\dell\Desktop\NTU\7207-Neural and Fuzzy Systems\K-Acc.png', dpi=300)
    plt.show()


def best_linear_fit(x, y):
    xbar = sum(x) / len(x)
    ybar = sum(y) / len(y)
    n = len(x)  # or len(y)

    numer = sum([xi * yi for xi, yi in zip(x, y)]) - n * xbar * ybar
    denum = sum([xi ** 2 for xi in x]) - n * xbar ** 2

    b = numer / denum
    a = ybar - b * xbar

    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b

def mse_neuron_plot():
    global K_Mse_result
    #MSE-Neuron Linear Fitting
    for i in range(75):
        for j in range(10):
            rbf_kmeans(26+i)
        list2.append(np.mean(K_Mse_result))
        K_Mse_result = []
    a, b = best_linear_fit(list_nnum, list2)
    yfit = [a + b * xi for xi in list_nnum]
    plt.plot(list_nnum, yfit)
    plt.scatter(list_nnum, list2, c='#7CFC00', s=10, label='RBF-KMeans', marker='o')
    plt.title("MSE-Neuron Linear Fitting", fontsize=22)
    plt.xlabel("Neuron Number", fontsize=12)
    plt.ylabel("MSE", fontsize=22)
    plt.axis([25, 102, 0.35, 0.55])
    plt.legend()
    plt.savefig(r'C:\Users\dell\Desktop\NTU\7207-Neural and Fuzzy Systems\MSE-Neuron.png', dpi=300)
    plt.show()


def acc_neuron_plot():
    global K_Acc_result
    #ACC-Neuron Linear Fitting
    for i in range(75):
        for j in range(10):
            rbf_kmeans(26+i)
        list2.append(np.mean(K_Acc_result))
        K_Acc_result = []
    a, b = best_linear_fit(list_nnum, list2)
    yfit = [a + b * xi for xi in list_nnum]
    plt.plot(list_nnum, yfit)
    plt.scatter(list_nnum, list2, c='#7CFC00', s=10, label='RBF-KMeans', marker='o')
    plt.title("Accuracy-Neuron Linear Fitting", fontsize=22)
    plt.xlabel("Neuron Number", fontsize=12)
    plt.ylabel("Accuracy", fontsize=22)
    plt.axis([25, 102, 0.8, 1])
    plt.legend()
    plt.savefig(r'C:\Users\dell\Desktop\NTU\7207-Neural and Fuzzy Systems\Acc-Neuron.png', dpi=300)
    plt.show()

def test_label(SVM_labelTest, RBF_labelTest, SOM_labelTest, KMeans_labelTest):


    SVMLabelTest = r'C:\Users\dell\PycharmProjects\7207Assignment\labelTest\SVMLabelTest.mat'
    RBFLabelTest = r'C:\Users\dell\PycharmProjects\7207Assignment\labelTest\RBFLabelTest.mat'
    SOMLabelTest = r'C:\Users\dell\PycharmProjects\7207Assignment\labelTest\SOMLabelTest.mat'
    KMeansLabelTest = r'C:\Users\dell\PycharmProjects\7207Assignment\labelTest\KMeansLabelTest.mat'

    scio.savemat(SVMLabelTest, {'SVM_labelTest': SVM_labelTest})
    scio.savemat(RBFLabelTest, {'RBF_labelTest': RBF_labelTest})
    scio.savemat(SOMLabelTest, {'RBF_labelTest': SOM_labelTest})
    scio.savemat(KMeansLabelTest, {'RBF_labelTest': KMeans_labelTest})

def overfit():
    global K_Acc_result
    global K_Acc_train
    for i in range(200):
        for j in range(10):
            rbf_kmeans(i+1)
        list2.append(np.mean(K_Acc_result))
        list3.append(np.mean(K_Acc_train))
        K_Acc_result = []
        K_Acc_train = []
        print("运行次数：", i)
    plt.plot(list_nnum_over, list2, c='#7CFC00', label='validation_set')
    plt.plot(list_nnum_over, list3, c='#FFA500', label='train_set')
    plt.title("Accuracy for train set and validation set--Neuron number", fontsize=12)
    plt.xlabel("Neuron Number", fontsize=12)
    plt.ylabel("Accuracy", fontsize=22)
    plt.axis([0, 201, 0.5, 1])
    plt.legend()
    plt.savefig(r'C:\Users\dell\Desktop\NTU\7207-Neural and Fuzzy Systems\Valid_train.png', dpi=300)
    plt.show()

if __name__ == "__main__":


    # SVM_labelTest = svm_classifier()
    # RBF_labelTest = rbf_random(8, 8)
    # KMeans_labelTest = rbf_kmeans(64)
    # SOM_labelTest = rbf_som(8, 8)
    # test_label(SVM_labelTest, RBF_labelTest, SOM_labelTest, KMeans_labelTest)
    #



    plot_mse()
    # plot_acc()
