import scipy.io as io
import numpy as np
from fire import Fire
from sklearn.linear_model import Perceptron
from sklearn.multiclass import OutputCodeClassifier


K = 10

def Summary(pre, gt):
    acc = {}
    ids = np.arange(len(gt))
    for c in range(K):
        c_ids = ids[gt==c]
        pre_c = pre[c_ids]
        acc[c] = (pre_c==c).sum() / len(pre_c)
    print(acc)

def main(filename="Samples.mat"):
    print("<---Samples  Test--->")
    sample_data = np.array([[1, 1], [2, 2], [2, 0], [0, 0], [1, 0], [0, 1]], dtype=float)
    sample_label = np.array([0, 0, 0, 1, 1, 1])
    model = Perceptron(penalty='l1')
    model.fit(sample_data, sample_label)
    print("Perceptron  W: {}".format(model.coef_))
    print("Perceptron  b: {}".format(model.intercept_))

    model = Perceptron(penalty='l2')
    model.fit(sample_data, sample_label)
    print("MSE W: {}".format(model.coef_))
    print("MSE  b: {}".format(model.intercept_))

    print('\n')
    print("<---MNIST Dataset--->")
    data = io.loadmat(filename)
    trn_data = data['TrainSamples']
    trn_label = data['TrainLabels'].reshape(-1)
    test_data = data['TestSamples']
    test_label = data['TestLabels'].reshape(-1)
    print("Perceptron Acc:")
    model = OutputCodeClassifier(Perceptron(penalty='l1'), code_size=10)
    model.fit(trn_data, trn_label)
    print("Total acc: {}".format(model.score(test_data, test_label)))
    Summary(model.predict(test_data), test_label)

    print("MSE Acc:")
    model = OutputCodeClassifier(Perceptron(penalty='l1'), code_size=10)
    model.fit(trn_data, trn_label)
    print("Total acc: {}".format(model.score(test_data, test_label)))
    Summary(model.predict(test_data), test_label)
    

if __name__ == "__main__":
    Fire(main)
