from vyper.user import Model
import pandas as pd
from sklearn.utils import shuffle
from vyper.user.explorer import DataProfiler
from openpyxl import Workbook

import math
import numpy as np
from vyper.utils.tools import StatisticalTools as st
from sklearn.preprocessing import OrdinalEncoder
import scipy as stats
from scipy.stats import chi2
from scipy.stats import uniform, randint
from sklearn.preprocessing import StandardScaler
from varclushi import VarClusHi
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)

from sklearn.cluster import DBSCAN, KMeans
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from sklearn.decomposition import PCA
from matplotlib.cm import viridis
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from yellowbrick.cluster import KElbowVisualizer


class dbs:
    def __init__(self, data: str):
        """
        Method to initialize the class and read the data
        :param data: str - path to the data
        """
        self.df = pd.read_csv(data)

    def get_shape(self):
        """
        Method to get the shape of the data
        :return: tuple - shape of the data
        """
        return self.df.shape

    def get_df(self):
        """
        Method to get the data
        :return: dataframe - data
        """
        return self.df

    def nearestneigh(self, n_neighbors: int):
        """
        Method to find the k-nearest neighbors for each data point in the dataframe
        :param n_neighbors: int - number of neighbors to consider
        :return: array - distances of the k-nearest neighbors
        """
        neigh = NearestNeighbors(n_neighbors=n_neighbors)
        nbrs = neigh.fit(self.df)
        distances, indices = nbrs.kneighbors(self.df)
        return distances

    @staticmethod
    def get_eps(distances):
        """
        Method to find the optimal epsilon value for the DBSCAN algorithm
        :param distances: array - distances of the k-nearest neighbors
        :return: float - optimal epsilon value
        """
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]
        i = np.arange(len(distances))
        kneec = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
        kneec.plot_knee()
        return kneec.knee_y

    def min_samples_selection(self, eps: float, multiple: int, maxK_per: float = 100, minK_per: float = 5,
                              rand_sample_prop: float = 0.2):
        """
        Method to find the optimal min_samples value for the DBSCAN algorithm
        :param eps: float - epsilon value
        :param multiple: int - no of values to skip between minK and maxK for each iteration
        :param maxK_per: float - maxK value in percentage of the total number of data points
        :param minK_per: float - minK value in percentage of the total number of data points
        :param rand_sample_prop: float - proportion of the data to be sampled
        :return: int - optimal min_samples value
        """
        sample_data = self.df.sample(frac=rand_sample_prop)
        minK = int((minK_per / 100) * (sample_data.shape[0]))
        maxK = int((maxK_per / 100) * (sample_data.shape[0]))
        nums = list(range(minK, maxK + 1, multiple))
        ss = []
        for i in list(range(minK, maxK + 1, multiple)):
            dbscan = DBSCAN(eps=eps, min_samples=i)
            dbscan.fit(sample_data)
            ss.append(silhouette_score(sample_data, dbscan.labels_))
        return (nums[np.argmax(ss)] / (sample_data.shape[0])) * 100

    def dbscan(self, eps: float, percent: float = 5):
        """
        Method to perform DBSCAN clustering
        :param eps: float - epsilon value
        :param percent: float - min_samples value in percentage of the total number of data points (minimum and default=5)
        :return: ndarray - cluster labels
        """
        dbscan = DBSCAN(eps=eps, min_samples=int((percent / 100) * (self.df.shape[0])))
        clusters = dbscan.fit_predict(self.df) + 1
        return clusters

    def pca(self, clusters, n_components: int = 2):
        """
        Method to perform PCA
        :param clusters: ndarray - cluster labels
        :param n_components: int - number of components
        :return: dataframe - principal components
        """
        pca = PCA(n_components=n_components)
        principalComponents = pca.fit_transform(self.df)
        principalComponents = pd.DataFrame(data=principalComponents, index=self.df.index, columns=['PC1', 'PC2'])
        principalComponents['cluster'] = clusters
        return principalComponents

    @staticmethod
    def scatter_plot(principalComponents):
        """
        Method to plot the scatter plot
        :param principalComponents: dataframe - principal components
        :return: scatter plot
        """
        plt.scatter(principalComponents['PC1'], principalComponents['PC2'], c=principalComponents['cluster'],
                    cmap=viridis)
        plt.colorbar()
        plt.show()

    def get_silhouette_score(self, clusters):
        """
        Method to get the silhouette score
        :param clusters: ndarray - cluster labels
        :return: float - silhouette score
        """
        return silhouette_score(self.df, clusters)

    def get_davies_bouldin_score(self, clusters):
        """
        Method to get the davies bouldin score
        :param clusters: ndarray - cluster labels
        :return: float - davies bouldin score
        """
        return davies_bouldin_score(self.df, clusters)

    def get_cluster_centers(self, clusters):
        """
        Method to get the cluster centers
        :param clusters: ndarray - cluster labels
        :return: dataframe - cluster centers
        """
        temp = self.df.copy()
        temp['cluster'] = clusters
        return temp.groupby(clusters).mean()

    def get_cluster_sizes(self, clusters):
        """
        Method to get the cluster sizes
        :param clusters: ndarray - cluster labels
        :return: dataframe - cluster sizes
        """
        temp = self.df.copy()
        temp['cluster'] = clusters
        return temp.groupby(clusters).size()
