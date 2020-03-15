from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from data_gen import create_texts
import argparse


COLOURS = [
    'b', 'g', 'r', 'c', 'm', 
    'y', 'k', 'aqua', 'indigo', 'crimson', 
    'olive', 'salmon', 'chocolate', 'orchid', 'coral',
    'cadetblue', 'peru', 'forestgreen', 'gold', 'navy', 
    'plum', 'fuchsia', 'coral', 'sienna']


def tsne(num_points=100, plot_dimension=3):
    assert plot_dimension in [2,3]

    # Getting data 
    word_vectors = create_texts(num_points)

    # Defining Model
    model = TSNE(n_components=plot_dimension, learning_rate=100)

    # Fitting Model
    transformed = model.fit_transform(word_vectors)

    # Plotting 2d t-Sne
    x_axis = transformed[:, 0]
    y_axis = transformed[:, 1]
    if plot_dimension == 3:
        z_axis = transformed[:, 2]

    if plot_dimension == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_axis, y_axis, z_axis)
    else:
        plt.scatter(x_axis, y_axis)
    plt.show()
    

def tsne_kmeans(num_points=100, plot_dimension=3, num_clusters = 12):
    assert plot_dimension in [2,3]

    # Getting data 
    word_vectors = create_texts(num_points)

    # Defining Model
    model = TSNE(n_components=plot_dimension, learning_rate=100)

    # Fitting Model
    transformed = model.fit_transform(word_vectors)

    # Plotting 2d t-Sne
    x_axis = transformed[:, 0]
    y_axis = transformed[:, 1]
    if plot_dimension == 3:
        z_axis = transformed[:, 2]

    # # Kmeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(transformed)
    # print(kmeans.labels_)
    # print(kmeans.cluster_centers_)

    if plot_dimension == 2:
        for i, label in enumerate(kmeans.labels_):
            plt.scatter(transformed[i, 0], transformed[i, 1], c=COLOURS[label])
        plt.xlabel("TSNE Dimension 1")
        plt.ylabel("TSNE Dimension 2")
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i, label in enumerate(kmeans.labels_):
            ax.scatter(transformed[i, 0], transformed[i, 1], transformed[i, 2], c=COLOURS[label])

        ax.set_xlabel("TSNE Dimension 1")
        ax.set_ylabel("TSNE Dimension 2")
        ax.set_zlabel("TSNE Dimension 3")


    plt.title("Kmeans Clustering (k={0}) of {1} TSNE-transformed Augmented Word Vectors".format(num_clusters, num_points))


    plt.show()
    


def dbscan(num_points=100):
    # Getting data 
    word_vectors = create_texts(num_points)

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

def pca(num_points=100):
    """
    Found out the categories (e.g. season) with the lowest number of elements were the main contributors to the main principle axes.
    In other words, in the current setup, the number of elements chosen for a particular category will affect the PCA.
    """
    
    # Getting data 
    word_vectors = create_texts(num_points)

    pca = PCA(n_components=5).fit(word_vectors)
    print(np.argmax(np.array(pca.components_), 1))


if __name__ == "__main__":
    # Parsing arguments ------------------------------------------
    parser = argparse.ArgumentParser("Unsupervised Learning Trial")
    parser.add_argument("-n", help="Number of datapoints generated.", type=int, default=100)

    args = parser.parse_args()

    ################################## SETUP #####################################
    tsne_kmeans(args.n, 2)