import numpy as np
import pandas as pd
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from scipy.stats import gaussian_kde


class My_GMM:
    def __init__(self, Data, K, weights=None, means=None, cov_matrixs=None):
        """
        GMM（高斯混合模型）类的构造函数
        :param Data: 训练数据
        :param K: 高斯分布的个数
        :param weights: 每个高斯分布的初始概率（权重）
        :param means: 高斯分布的均值向量
        :param cov_matrixs: 高斯分布的协方差矩阵集合
        """
        self.Data = Data
        self.K = K
        if weights is not None:
            self.weights = weights
        else:
            # 随机初始化权重
            self.weights = np.random.rand(self.K)
            self.weights /= np.sum(self.weights)  # 归一化

        # row = np.shape(self.Data)[0]
        col = np.shape(self.Data)[1]

        if means is not None:
            self.means = means
        else:
            # 随机初始化均值向量(对每个高斯分布)
            self.means = []
            for i in range(self.K):
                mean = np.random.rand(col)
                # mean = mean / np.sum(mean)        # 归一化
                self.means.append(mean)

        if cov_matrixs is not None:
            self.covars = cov_matrixs
        else:
            # 随机初始化协方差矩阵(对每个高斯分布)
            self.covars = []
            for i in range(self.K):
                cov = np.random.rand(col, col)
                # cov = cov / np.sum(cov)                    # 归一化
                self.covars.append(cov)  # cov是np.array,但是self.covars是list

    def clamp(self, x):
        x = np.clip(x, a_max=1e3, a_min=-1e3)
        #x[np.abs(x) < 1e-4] = np.sign(x[np.abs(x) < 1e-4]) * 1e-4
        return x

    def Gaussian(self, x, mean, cov):
        """
        高斯分布概率密度函数
        :param x: 输入数据
        :param mean: 均值数组
        :param cov: 协方差矩阵
        :return: x的概率
        """

        dim = np.shape(cov)[0]
        # cov的行列式为零时的措施

        if np.isnan(np.sum(cov)):
            cov = np.eye(dim) * 0.01

#        covdet = np.linalg.det(cov + np.eye(dim) * 0.01)
#        covinv = np.linalg.inv(cov + np.eye(dim) * 0.01)
        covdet = np.linalg.det(cov)
        covinv = np.linalg.inv(cov)
        xdiff = (x - mean).reshape((1, dim))
        # 概率密度

        # print(cov)

        a = np.power(2 * np.pi, dim)
        a = np.power(a * np.abs(covdet), 0.5)
        b = xdiff.dot(covinv).dot(xdiff.T)
        b = self.clamp(b)
        b = np.exp(-0.5 * b)[0][0]
        prob = 1.0 / (a)
        prob = prob * b
#        prob = 1.0 / (np.power(np.power(2 * np.pi, dim) * np.abs(covdet), 0.5)) * \
#               np.exp(-0.5 * xdiff.dot(covinv).dot(xdiff.T))[0][0]

        return prob

    def EM(self):
        """
        训练阶段 , EM算法
        :return:
        # self.posibility 表示第j个观测数据属于第k个gmm的概率
        # self.prediction 表示第i个数据的类别(取argmax得到的)
        """
        log_likelyhood = 0
        old_log_likelyhood = 1
        len, dim = np.shape(self.Data)
        # gamma表示第j个观测数据属于第k个gmm的概率
        gammas = [np.zeros(self.K) for i in range(len)]
        while np.abs(log_likelyhood - old_log_likelyhood) > 0.0001:
            old_log_likelyhood = log_likelyhood
            # E-step
            for n in range(len):
                # 计算后验概率
                respons = [self.weights[k] * self.Gaussian(self.Data[n], self.means[k], self.covars[k]) for k in
                           range(self.K)]
                respons = np.array(respons) + 1e-6  # 控制精度，防止溢出
                sum_respons = np.sum(respons)
                sum_respons = 1e12 if sum_respons == np.inf else sum_respons
                gammas[n] = respons / sum_respons

            # M-step
            for k in range(self.K):
                # N_k表示N个样本中有多少属于第k个高斯
                N_k = np.sum([gammas[n][k] for n in range(len)])
                # 更新每个高斯分布的概率
                self.weights[k] = 1.0 * N_k / len
                # 更新高斯分布的均值
                self.means[k] = (1.0 / N_k) * np.sum([gammas[n][k] * self.Data[n] for n in range(len)], axis=0)
                xdiffs = self.Data - self.means[k]
                # 更新高斯分布的协方差矩阵
                self.covars[k] = (1.0 / N_k) * np.sum(
                    [gammas[n][k] * xdiffs[n].reshape((dim, 1)).dot(xdiffs[n].reshape((1, dim))) for n in range(len)],
                    axis=0)
            log_likelyhood = []
            for n in range(len):
                tmp = [np.sum(self.weights[k] * self.Gaussian(self.Data[n], self.means[k], self.covars[k])) for k in
                       range(self.K)]
                tmp = np.log(np.array(tmp) + 1e-6)  # 控制精度，防止溢出
                log_likelyhood.append(list(tmp))
            log_likelyhood = np.sum(log_likelyhood)
        for i in range(len):
            gammas[i] = gammas[i] / np.sum(gammas[i])

        self.posibility = gammas
        self.prediction = [np.argmax(gammas[i]) for i in range(len)]

    def test(self, test_data):
        """
        测试阶段 , 使用EM算法的E-step即可 , 此时GMM参数都训出来了
        :return:

        """
        len, dim = np.shape(test_data)
        # gamma表示第j个观测数据属于第k个gmm的概率
        gammas = [np.zeros(self.K) for i in range(len)]

        # E-step
        for n in range(len):
            # 计算后验概率
            respons = [self.weights[k] * self.Gaussian(test_data[n], self.means[k], self.covars[k]) for k in
                       range(self.K)]
            respons = np.array(respons)
            gammas[n] = respons

        # 对每个分量的概率求和，即属于该GMM的概率
        test_acc = [np.sum(gammas[i]) for i in range(len)]
        return test_acc


def main_part1():
    train_data1 = np.loadtxt('Train1.csv', dtype='object')
    train_data1 = np.array([[float(d) for d in data.split(",")] for data in train_data1])  # string转float

    # GMM模型
    K1 = 2
    gmm1 = My_GMM(train_data1, K1)
    gmm1.EM()
#    print('-----gmm1 parameter-----')
#    print(gmm1.weights)
#    print(gmm1.means)
#    print(gmm1.covars)
    # print(gmm1.posibility)
    # print(gmm1.prediction)

    train_data2 = np.loadtxt('Train2.csv', dtype='object')
    train_data2 = np.array([[float(d) for d in data.split(",")] for data in train_data2])  # string转float

    # GMM模型
    K2 = 2
    gmm2 = My_GMM(train_data2, K2)
    gmm2.EM()
#    print('\n-----gmm2 parameter-----')
#    print(gmm2.weights)
#    print(gmm2.means)
#    print(gmm2.covars)
    # print(gmm2.posibility)
    # print(gmm2.prediction)

    print('\n-----test phrase1-----')

    test_data1 = np.loadtxt('Test1.csv', dtype='object')
    test_data1 = np.array([[float(d) for d in data.split(",")] for data in test_data1])  # string转float
    label1 = np.zeros(test_data1.shape[0])

    test_data1_gmm1 = gmm1.test(test_data1)
    test_data1_gmm2 = gmm2.test(test_data1)

#    print(test_data1_gmm1)
#    print(test_data1_gmm2)

    pre1 = np.array([(0 if test_data1_gmm1[i] > test_data1_gmm2[i] else 1) for i in range(test_data1.shape[0])])
    print("test1正确率为：\n", accuracy_score(label1.tolist(), pre1.tolist()))

    print('\n-----test phrase2-----')

    test_data2 = np.loadtxt('Test2.csv', dtype='object')
    test_data2 = np.array([[float(d) for d in data.split(",")] for data in test_data2])  # string转float
    label2 = np.ones(test_data2.shape[0])

    test_data2_gmm1 = gmm1.test(test_data2)
    test_data2_gmm2 = gmm2.test(test_data2)

#    print(test_data2_gmm1)
#    print(test_data2_gmm2)

    pre2 = np.array([(0 if test_data2_gmm1[i] > test_data2_gmm2[i] else 1) for i in range(test_data2.shape[0])])
    print("test2正确率为：\n", accuracy_score(label2.tolist(), pre2.tolist()))


def main_part2():
    train_samples = np.loadtxt('TrainSamples.csv', dtype='object')
    train_samples = np.array([[float(d) for d in data.split(",")] for data in train_samples])  # string转float

    train_lbs = np.loadtxt('TrainLabels.csv', dtype='object')
    train_lbs = np.array([[float(d) for d in data.split(",")] for data in train_lbs])  # string转float

    test_samples = np.loadtxt('TestSamples.csv', dtype='object')
    test_samples = np.array([[float(d) for d in data.split(",")] for data in test_samples])  # string转float

    test_lbs = np.loadtxt('TestLabels.csv', dtype='object')
    test_lbs = np.array([[float(d) for d in data.split(",")] for data in test_lbs])  # string转float

    # 数据转换
    train_data = []
    train_labels = []

    for i in range(10):
        train_data.append(train_samples[train_lbs.squeeze() == i])
        train_labels.append(np.zeros(len(train_data[-1])) + i)

    for k in range(1, 6):
        gmms = []
        # train phase
        for i in range(10):
            gmm = My_GMM(train_data[i], k)
            gmm.EM()
            gmms.append(gmm)

        # test phase
        test_data_gmms = []

        for i in range(10):
            test_data_gmms.append(gmms[i].test(test_samples))

        pre = []
        for j in range(test_samples.shape[0]):
            # print([test_data_gmms[i][j] for i in range(10)])
            pre.append(np.argmax([test_data_gmms[i][j] for i in range(10)]))

#        print(pre)

        # print(np.array(test_lbs).squeeze())
        # print(np.array(pre))
        print("test正确率为：\n", accuracy_score(np.array(test_lbs).squeeze(), np.array(pre)))


main_part2()
