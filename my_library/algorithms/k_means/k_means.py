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


class Kmeans:

    def __init__(self, data):
        self.df = pd.read_csv(data)

    def Kmeans_elbow_plot(self, minK, maxK):
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
            # mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
            #                                'euclidean'), axis=1)) / X.shape[0]
            # mapping2[k] = kmeanModel.inertia_

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

    def kmeans_minClusterSize(self, cluster_num):
        minSize = 0  # (means 0*100)
        maxSize = int(len(self.df) / (100 * cluster_num)) + 1
        Size = range(minSize, maxSize)
        kmeanModel_minSize_list = []
        kmeanModel_minSize_out_list = []
        distortions = []
        inertias = []
        # mapping1 = {}
        # mapping2 = {}
        for k in Size:
            print("*** size: " + str(100 * k), end=", ")
            # Building and fitting the model
            kmeanModel_minSize = KMeansConstrained(
                n_clusters=cluster_num,
                size_min=100 * k,
                random_state=0)
            kmeanModel_minSize_out_list.append(kmeanModel_minSize.fit_predict(self.df))
            kmeanModel_minSize_list.append(kmeanModel_minSize)
            distortions.append(
                sum(np.min(cdist(self.df, kmeanModel_minSize.cluster_centers_, 'euclidean'), axis=1)) / self.df.shape[
                    0])
            inertias.append(kmeanModel_minSize.inertia_)
            # mapping1[k] = sum(np.min(cdist(self.df, kmeanModel_minSize.cluster_centers_,'euclidean'), axis=1)) / self.df.shape[0]
            # mapping2[k] = kmeanModel_minSize.inertia_

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
        return kmeanModel_minSize_list
