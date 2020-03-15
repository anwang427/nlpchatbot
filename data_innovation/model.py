from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from data_gen import create_texts



def tsne():
    # Getting data 
    word_vectors = create_texts(100)

    # Defining Model
    model = TSNE(n_components=3, learning_rate=100)

    # Fitting Model
    transformed = model.fit_transform(word_vectors)

    # Plotting 2d t-Sne
    x_axis = transformed[:, 0]
    y_axis = transformed[:, 1]
    z_axis = transformed[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_axis, y_axis, z_axis)
    plt.show()


def dbscan():
    # Getting data 
    word_vectors = create_texts(100)

    # Fitting DBSCAN model
    model = DBSCAN()
    model.fit(word_vectors)

    # PCA for visualization purposes
    pca = PCA(n_components=2).fit(word_vectors)
    pca_2d = pca.transform(word_vectors)

    # print(dbscan.labels_)
    for i in range(0, pca_2d.shape[0]):
        if dbscan.labels_[i] == 0:
            c1 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
        elif dbscan.labels_[i] == 1:
            c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='o')
        elif dbscan.labels_[i] == -1:
            c3 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='*')

    plt.title('')
    plt.show()

def pca():
    """
    Found out the categories (e.g. season) with the lowest number of elements were the main contributors to the main principle axes.
    In other words, in the current setup, the number of elements chosen for a particular category will affect the PCA.
    """
    
    # Getting data 
    word_vectors = create_texts(100)

    pca = PCA(n_components=5).fit(word_vectors)
    print(np.argmax(np.array(pca.components_), 1))


pca()