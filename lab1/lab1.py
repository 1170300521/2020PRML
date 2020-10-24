import scipy.io as io
import pandas as pd
import numpy as np
from numpy.linalg import norm as distance
import matplotlib.pyplot as plt
from fire import Fire


# simple samples to testing K-means algs
samples = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 1], [1, 2], [2, 2], [3, 2], [6, 6], [7, 6], [8,6], [7, 7], [8, 7], [9, 7], [7, 8], [8, 8], [9, 8], [8, 9], [9, 9]], dtype=float)
colors = ['r', 'b', 'g']

def K_means(data, centroids, K):
    clusters = [[] for i in range(K)]
    for i, d in enumerate(data):
        dists = []
        for c in centroids:
            dists.append(distance(c - d))
        dists = np.array(dists)
        idx = np.argsort(dists)[0]
        clusters[idx].append(i)
    centroids = []
    for c in clusters:
        centroids.append(data[c].mean(axis=0))
    return clusters, np.array(centroids)

def Summary(clusters, labels, K=10):
    stat = {i:[] for i in range(K)}
    for i in range(K):
        for c in clusters:
            stat[i].append((labels[c]==i).sum())
    frame = pd.DataFrame(stat)
    print(frame)

def main(filename="ClusterSamples.mat", K1=2, K2=10, T1=5, T2=1000):
    # cluster simple samples 
    print("<--------Simple Samples Cluster-------->")
    centroids = samples[np.random.choice(len(samples), K1)]
    for i in range(T1):
        clusters, centroids = K_means(samples, centroids, K1)
        print(clusters)
    # plot the cluster results
    plt.figure()
    for i, c in enumerate(clusters):
        single = samples[c]
        plt.scatter(single[:,0], single[:,1], color=colors[i])
    plt.show()

    print("<------ClusterSamples.mat Cluster------>")
    # read raw data from .mat file, which including vectors and responding labels
    raw_data = io.loadmat(filename)
    vectors = raw_data['ClusterSamples']
    labels = raw_data['SampleLabels']
    centroids = vectors[np.random.choice(len(labels), K2)]
    
    for i in range(T2):
        clusters, centroids = K_means(vectors, centroids, K2)
    # print results
    for c in clusters:
        print(len(c))
    Summary(clusters, labels)


if __name__ == "__main__":
    Fire(main)
