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
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


class Hierarchical:
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

    def dendogram(self):
        """
        Method to plot the dendogram
        """
        plt.figure(figsize=(10, 7))
        plt.title("Dendogram")
        shc.dendrogram(shc.linkage(self.df, method='ward'))
        plt.show()

    @staticmethod
    def silhouette(estimator, df, metric: str = 'euclidean'):
        """
        Method to calculate the silhouette score
        :param estimator: object - estimator
        :param df: dataframe - data
        :param metric: str - metric
        :return: float - silhouette score
        """
        # print(self.df.shape)
        labels = estimator.fit_predict(df)
        score = silhouette_score(df, labels, metric=metric)
        return score

    @staticmethod
    def get_training_history(training_history: dict):
        """
        Method to get the training history
        :param training_history: dict - training history
        :return:
        """
        return pd.DataFrame.from_dict(training_history)

    def gridSearchCV_hierarchical(self, grid: dict, cv: int = 5):
        """
        Method to perform grid search cross validation
        :param grid: dict - grid
        :param cv: int - number of folds
        :return: tuple - best score, best parameters, training history
        """
        hierarchical = AgglomerativeClustering()
        grid_search = GridSearchCV(estimator=hierarchical, param_grid=grid, cv=cv, scoring=self.silhouette)
        grid_search.fit(self.df)
        training_history = grid_search.cv_results_
        return grid_search.best_score_, grid_search.best_params_, training_history

    def randomizedSearchCV_hierarchical(self, grid: dict, cv: int = 5, n_iter: int = 10, rand_sample_prop: float = 0.2):
        """
        Method to perform randomized search cross validation
        :param grid: dict - grid
        :param cv: int - number of folds
        :param n_iter: int - number of iterations
        :param rand_sample_prop: float - random sample proportion
        :return: tuple - best score, best parameters, training history
        """
        # fd = self.create_folds(cv)
        sample_data = self.df.sample(frac=rand_sample_prop)
        hierarchical = AgglomerativeClustering()
        random_search = RandomizedSearchCV(estimator=hierarchical, param_distributions=grid, cv=cv,
                                           scoring=self.silhouette, n_iter=n_iter)
        random_search.fit(sample_data)
        training_history = random_search.cv_results_
        return random_search.best_score_, random_search.best_params_, training_history

    def hierarchical(self, n_clusters: int, linkage: str, affinity: str):
        """
        Method to perform hierarchical clustering
        :param n_clusters: int - number of clusters
        :param linkage: str - linkage
        :param affinity: str - affinity
        :return: ndarray - clusters
        """
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, linkage=linkage)
        clusters = hierarchical.fit_predict(self.df)
        return clusters

    def pca(self, clusters, n_components: int = 2):
        """
        Method to perform PCA
        :param clusters: ndarray - clusters
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
        :param clusters: ndarray - clusters
        :return: float - silhouette score
        """
        return silhouette_score(self.df, clusters)

    def get_davies_bouldin_score(self, clusters):
        """
        Method to get the davies bouldin score
        :param clusters: ndarray - clusters
        :return: float - davies bouldin score
        """
        return davies_bouldin_score(self.df, clusters)

    def get_cluster_centers(self, clusters):
        """
        Method to get the cluster centers
        :param clusters: ndarray - clusters
        :return: dataframe - cluster centers
        """
        temp = self.df.copy()
        temp['cluster'] = clusters
        return temp.groupby(clusters).mean()

    def get_cluster_sizes(self, clusters):
        """
        Method to get the cluster sizes
        :param clusters: ndarray - clusters
        :return: dataframe - cluster sizes
        """
        temp = self.df.copy()
        temp['cluster'] = clusters
        return temp.groupby(clusters).size()
