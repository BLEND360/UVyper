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

from sklearn.cluster import Birch
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import fbeta_score, make_scorer, silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from matplotlib.cm import viridis


class birch:
    def __init__(self, data: str):
        """
        Method to initialize the class and read the data
        :param data: str - path to the data
        """
        self.df = pd.read_csv(data)
        # self.df = self.df.iloc[:50000, :]

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

    @staticmethod
    def silhouette(estimator, df, metric: str = 'euclidean'):
        """
         Method to calculate the silhouette score.
         :param estimator: object - clustering algorithm that implements fit_predict method
         :param df: DataFrame - data
         :param metric: str - metric to use for calculating the distance between points
         :return: float - silhouette score
         """
        labels = estimator.fit_predict(df)
        score = silhouette_score(df, labels, metric=metric)
        return score

    @staticmethod
    def get_training_history(training_history: dict):
        """
        Method to get the training history
        :param training_history: dict - training history
        :return: dataframe - training history
        """
        return pd.DataFrame.from_dict(training_history)

    def gridSearchCV_birch(self, grid: dict, cv: int = 5, rand_sample_prop: float = 0.2):
        """
        Method to perform grid search cross validation
        :param grid: dict - grid of parameters
        :param cv: int - number of folds
        :param rand_sample_prop: float - proportion of data to sample, default is 0.2
        :return: tuple - best score, best parameters, training history
        """
        sample_data = self.df.sample(frac=rand_sample_prop)
        birch = Birch()
        grid_search = GridSearchCV(estimator=birch, param_grid=grid, cv=cv, scoring=self.silhouette)
        grid_search.fit(sample_data)
        # grid_search.fit(self.df)
        return grid_search.best_score_, grid_search.best_params_, grid_search.cv_results_

    def randomizedSearchCV_birch(self, grid: dict, cv: int = 5, n_iter: int = 10, rand_sample_prop: float = 0.2):
        """
        Method to perform randomized search cross validation
        :param grid: dict - grid of parameters
        :param cv: int - number of folds
        :param n_iter: int - number of iterations
        :param rand_sample_prop: float - proportion of data to sample, default is 0.2
        :return: tuple - best score, best parameters, training history
        """
        sample_data = self.df.sample(frac=rand_sample_prop)
        birch = Birch()
        random_search = RandomizedSearchCV(estimator=birch, param_distributions=grid, cv=cv, scoring=self.silhouette,
                                           n_iter=n_iter)
        random_search.fit(sample_data)
        return random_search.best_score_, random_search.best_params_, random_search.cv_results_

    def birch(self, branching_factor: int, n_clusters: int, threshold: float):
        """
        Method to perform birch clustering
        :param branching_factor: int - branching factor
        :param n_clusters: int - number of clusters
        :param threshold: float - threshold
        :return: ndarray - cluster labels
        """
        birch = Birch(branching_factor=branching_factor, n_clusters=n_clusters, threshold=threshold)
        clusters = birch.fit_predict(self.df)
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
