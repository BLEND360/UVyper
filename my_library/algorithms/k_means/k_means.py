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


class Kmeans:

    def __init__(self, data):  # reading the data
        self.df = pd.read_csv(data)

    def Kmeans_elbow_plot(self, minK, maxK):  # elbow plot for kmeans using distortion and inertia
        distortions = []
        inertias = []
        K = range(minK, maxK)
        X = self.df

        for k in K:
            # Building and fitting the model
            kmeanModel = KMeans(n_clusters=k).fit(X)
            kmeanModel.fit(X)
            distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                                'euclidean'), axis=1)) / X.shape[0])
            inertias.append(kmeanModel.inertia_)

        plt.plot(K, distortions, 'bx-')
        plt.xlabel('Values of K')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method using Distortion')
        plt.show()

        plt.plot(K, inertias, 'bx-')
        plt.xlabel('Values of K')
        plt.ylabel('Inertia')
        plt.title('The Elbow Method using Inertia')
        plt.show()

    def kmeans_minClusterSize(self, cluster_num, multiple):

        # The 'multiple' parameter is used in this function to
        # calculate the minimum and maximum possible cluster sizes based on the size of the dataset and the desired
        # number of clusters. Specifically, the maximum cluster size is calculated as a fraction of the dataset size,
        # where the fraction is determined by the value of 'multiple' (i.e., maxSize = int(len(self.df) / (multiple *
        # cluster_num)) + 1). By adjusting the value of 'multiple', the user can control the range of cluster sizes
        # that are considered in the elbow plot. For example, setting 'multiple' to a higher value will result in a
        # smaller range of cluster sizes being considered, whereas setting 'multiple' to a lower value will result in
        # a larger range of cluster sizes being considered. Ultimately, the choice of 'multiple' will depend on the
        # specific dataset and the user's goals for clustering.

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

        # The 'multiple' parameter is used in this function to
        # calculate the minimum and maximum possible cluster sizes based on the size of the dataset and the desired
        # number of clusters. Specifically, the maximum cluster size is calculated as a fraction of the dataset size,
        # where the fraction is determined by the value of 'multiple' (i.e., maxSize = int(len(self.df) / (multiple *
        # cluster_num)) + 1). By adjusting the value of 'multiple', the user can control the range of cluster sizes
        # that are considered in the elbow plot. For example, setting 'multiple' to a higher value will result in a
        # smaller range of cluster sizes being considered, whereas setting 'multiple' to a lower value will result in
        # a larger range of cluster sizes being considered. Ultimately, the choice of 'multiple' will depend on the
        # specific dataset and the user's goals for clustering.

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

    def kmeans(self, n_clusters, min_size, max_size):
        kmeanModel = KMeansConstrained(
            n_clusters=n_clusters,
            size_min=min_size,
            size_max=max_size,
            random_state=42)
        clusters = kmeanModel.fit_predict(self.df)
        return clusters

    def pca(self, clusters, n_components=2):
        pca = PCA(n_components=n_components)
        principalComponents = pca.fit_transform(self.df)
        principalComponents = pd.DataFrame(data=principalComponents, index=self.df.index, columns=['PC1', 'PC2'])
        principalComponents['cluster'] = clusters
        return principalComponents

    @staticmethod
    def scatter_plot(principalComponents):
        plt.scatter(principalComponents['PC1'], principalComponents['PC2'], c=principalComponents['cluster'])
        plt.show()
