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
from sklearn.preprocessing import StandardScaler
from varclushi import VarClusHi
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from k_means_constrained import KMeansConstrained
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from kneed import KneeLocator
from yellowbrick.cluster import KElbowVisualizer
from matplotlib.cm import viridis


class Kmeans:

    def __init__(self, data):
        """
        __init__ method to read the data
        :param data: str - Path to the CSV file containing the data to be preprocessed.
        """
        self.df = pd.read_csv(data)

    def get_data(self):
        """
        Method to return the data
        :return: dataframe
        """
        return self.df

    def elbow(self, minK, maxK, metric='distortion'):
        """
        Method to identify the optimal number of clusters (k) for a given dataset using the elbow method.
        :param minK: int - The minimum number of clusters to consider.
        :param maxK: int - The maximum number of clusters to consider.
        :param metric: str - optional (default='distortion'). The metric used to quantify the quality of clustering. Possible options include distortion, silhouette, calinski_harabasz, davies_bouldin, and others.
        :return: int - The optimal number of clusters (k).
        """

        X = self.df
        model = KMeans()
        visualizer = KElbowVisualizer(model, k=(minK, maxK), metric=metric, timings=False)
        visualizer.fit(X)
        visualizer.show()
        return visualizer.elbow_value_

    def kmeans_minClusterSize(self, cluster_num, multiple):
        """
        Method to find the minimum number of points in cluster.
        :param cluster_num: int - The number of clusters to consider.
        :param multiple: int - int - Number of values to skip between minimum and maximum for each iteration.
        """

        minSize = 0
        maxSize = int(len(self.df) / (multiple * cluster_num)) + 1
        Size = range(minSize, maxSize)
        kmeanModel_minSize_list = []
        kmeanModel_minSize_out_list = []
        distortions = []
        inertias = []

        for k in Size:
            print("*** size: " + str(multiple * k), end=", ")
            # Building and fitting the model
            kmeanModel_minSize = KMeansConstrained(
                n_clusters=cluster_num,
                size_min=multiple * k,
                random_state=0)
            kmeanModel_minSize_out_list.append(kmeanModel_minSize.fit_predict(self.df))
            kmeanModel_minSize_list.append(kmeanModel_minSize)
            distortions.append(
                sum(np.min(cdist(self.df, kmeanModel_minSize.cluster_centers_, 'euclidean'), axis=1)) / self.df.shape[
                    0])
            inertias.append(kmeanModel_minSize.inertia_)

        plt.plot(Size, distortions, 'bx-')
        plt.xlabel('Values of Size')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method using Distortion for Minimum Size Selection,Cluster_Num=' + str(cluster_num))
        plt.show()

        plt.plot(Size, inertias, 'bx-')
        plt.xlabel('Values of Size')
        plt.ylabel('Inertia')
        plt.title('The Elbow Method using Inertia for Minimum Size Selection,Cluster_Num=' + str(cluster_num))
        plt.show()

    def kmeans_maxClusterSize(self, cluster_num, multiple):
        """
        Method to find the maximum number of points in cluster.
        :param cluster_num: int - The number of clusters to consider.
        :param multiple: int - Number of values to skip between minimum and maximum for each iteration.
        """

        minSize = int(len(self.df) / (multiple * cluster_num)) + 1
        maxSize = int(len(self.df) / multiple)
        Size = range(minSize, maxSize)
        kmeanModel_maxSize_list = []
        kmeanModel_maxSize_out_list = []
        distortions = []
        inertias = []
        for k in Size:
            print("*** size: " + str(multiple * k), end=", ")
            # Building and fitting the model
            kmeanModel_maxSize = KMeansConstrained(
                n_clusters=cluster_num,
                size_max=multiple * k,
                random_state=42)
            kmeanModel_maxSize_out_list.append(kmeanModel_maxSize.fit_predict(self.df))
            kmeanModel_maxSize_list.append(kmeanModel_maxSize)
            distortions.append(
                sum(np.min(cdist(self.df, kmeanModel_maxSize.cluster_centers_, 'euclidean'), axis=1)) / self.df.shape[
                    0])
            inertias.append(kmeanModel_maxSize.inertia_)

        plt.plot(Size, distortions, 'bx-')
        plt.xlabel('Values of Size')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method using Distortion for Maximum Size Selection,Cluster_Num=' + str(cluster_num))
        plt.show()

        plt.plot(Size, inertias, 'bx-')
        plt.xlabel('Values of Size')
        plt.ylabel('Inertia')
        plt.title('The Elbow Method using Inertia for Maximum Size Selection,Cluster_Num=' + str(cluster_num))
        plt.show()

    # added the below three functions to the class
    def kmeans(self, n_clusters, min_size, max_size):
        """
        Method to perform k-means clustering.
        :param n_clusters: int - The number of clusters to consider.
        :param min_size: int - The minimum number of points in a cluster.
        :param max_size: int - The maximum number of points in a cluster.
        :return: list - The cluster labels.
        """
        kmeanModel = KMeansConstrained(
            n_clusters=n_clusters,
            size_min=min_size,
            size_max=max_size,
            random_state=42)
        clusters = kmeanModel.fit_predict(self.df)
        return clusters

    def pca(self, clusters, n_components=2):
        """
        Method to perform PCA.
        :param clusters: list - The cluster labels.
        :param n_components: int - The number of components to consider.
        :return: dataframe - The PCA results.
        """
        pca = PCA(n_components=n_components)
        principalComponents = pca.fit_transform(self.df)
        principalComponents = pd.DataFrame(data=principalComponents, index=self.df.index, columns=['PC1', 'PC2'])
        principalComponents['cluster'] = clusters
        return principalComponents

    @staticmethod
    def scatter_plot(principalComponents):
        """
        Method to plot the PCA results.
        :param principalComponents: dataframe - The PCA results.
        :return: Scatter Plot
        """
        plt.scatter(principalComponents['PC1'], principalComponents['PC2'], c=principalComponents['cluster'],
                    cmap=viridis)
        plt.colorbar()
        plt.show()

    def get_silhouette_score(self, clusters):
        """
        Method to calculate the silhouette score.
        :param clusters: list - The cluster labels.
        :return: float - The silhouette score.
        """
        return silhouette_score(self.df, clusters)

    def get_davies_bouldin_score(self, clusters):
        """
        Method to calculate the Davies-Bouldin score.
        :param clusters: list - The cluster labels.
        :return: float - The Davies-Bouldin score.
        """
        return davies_bouldin_score(self.df, clusters)

    def get_cluster_centers(self, clusters):
        """
        Method to calculate the cluster centers.
        :param clusters: list - The cluster labels.
        :return: DataFrame - The cluster centers.
        """
        temp = self.df.copy()
        temp['cluster'] = clusters
        return temp.groupby(clusters).mean()

    def get_cluster_sizes(self, clusters):
        """
        Method to calculate the cluster sizes.
        :param clusters: list - The cluster labels.
        :return: DataFrame - The cluster sizes.
        """
        temp = self.df.copy()
        temp['cluster'] = clusters
        return temp.groupby(clusters).size()
