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

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from matplotlib.cm import viridis
import pickle
import os


class GMM:
    def __init__(self, data):
        """
        Method to initialize the GMM class by reading in data from a CSV file.
        :param data: str - path to the CSV file
        """
        self.df = pd.read_csv(data)

    def get_shape(self):
        """
        Method to get the shape of the data.
        :return: tuple - shape of the data
        """
        return self.df.shape

    def get_df(self):
        """
        Method to get the data.
        :return:
        """
        return self.df

    @staticmethod
    def silhouette(estimator, df, metric='euclidean'):
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
    def get_training_history(training_history):
        """
        Method to get the training history.
        :param training_history: dict - training history
        :return: dataframe - training history
        """
        return pd.DataFrame.from_dict(training_history)

    def gridSearchCV_gmm(self, grid, cv=5, rand_sample_prop=0.2):
        """
        Method to perform grid search cross validation on the GMM algorithm.
        :param grid: dic - dictionary of parameters to search over
        :param cv: int - number of folds
        :param rand_sample_prop: float - proportion of data to sample
        :return: tuple - best score, the best parameters, and cross validation results
        """
        sample_data = self.df.sample(frac=rand_sample_prop)
        gmm = GaussianMixture()
        grid_search = GridSearchCV(estimator=gmm, param_grid=grid, cv=cv, scoring=self.silhouette)
        grid_search.fit(sample_data)
        # grid_search.fit(self.df)
        return grid_search.best_score_, grid_search.best_params_, grid_search.cv_results_

    def randomizedSearchCV_gmm(self, grid, cv=5, n_iter=10, rand_sample_prop=0.2):
        """
        Method to perform randomized search cross validation on the GMM algorithm.
        :param grid: dict - dictionary of parameters to search over
        :param cv: int - number of folds
        :param n_iter: int - number of iterations
        :param rand_sample_prop: float - proportion of data to sample
        :return: tuple - best score, the best parameters, and cross validation results
        """
        sample_data = self.df.sample(frac=rand_sample_prop)
        gmm = GaussianMixture()
        random_search = RandomizedSearchCV(estimator=gmm, param_distributions=grid, cv=cv, scoring=self.silhouette,
                                           n_iter=n_iter)
        random_search.fit(sample_data)
        return random_search.best_score_, random_search.best_params_, random_search.cv_results_

    def gmm(self, n_components, covariance_type, init_params, rand_sample_prop=1.0):
        """
        Method to perform GMM clustering.
        :param n_components: int - number of clusters
        :param covariance_type: str - covariance type
        :param init_params: str - initialization method
        :param rand_sample_prop: float - proportion of data to sample
        :return: numpy array - cluster labels
        """
        sample_data = self.df.sample(frac=rand_sample_prop, random_state=42)
        gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, init_params=init_params)
        gmm.fit(sample_data)
        with open('gmmModel.pkl', 'wb') as f:
            pickle.dump(gmm, f)
        with open('gmmModel.pkl', 'rb') as f:
            gmm = pickle.load(f)
        clusters = gmm.predict(self.df)
        return clusters

    def gmm_model_create(self, n_components, covariance_type, init_params, rand_sample_prop=1.0):
        """
        Method to create pickle file of gmm model.
        :param n_components: int - number of clusters
        :param covariance_type: str - covariance type
        :param init_params: str - initialization method
        :param rand_sample_prop: float - proportion of data to sample
        :return: numpy array - cluster labels
        """
        sample_data = self.df.sample(frac=rand_sample_prop, random_state=42)
        gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, init_params=init_params)
        gmm.fit(sample_data)
        with open('gmmModel.pkl', 'wb') as f:
            pickle.dump(gmm, f)

    def gmm_model_read(self, filename):
        """
        Method to load the pickle file of gmm model.
        :param filename: str - path to the pickle file
        :return: ndarray - cluster labels
        """
        with open(filename, 'rb') as f:
            gmm = pickle.load(f)
        clusters = gmm.predict(self.df)
        return clusters

    def pca(self, clusters, n_components=2):
        """
        Method to perform PCA.
        :param clusters: numpy array - cluster labels
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
        Method to plot the principal components.
        :param principalComponents: dataframe - principal components
        :return: scatter plot
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

    @staticmethod
    def get_cluster_distribution(clusters):
        """
        Method to calculate the distribution of clusters.
        :param clusters: ndarray - The cluster labels.
        :return: dataframe - The distribution of clusters.
        """
        df = pd.DataFrame()
        df['cluster'] = sorted(set(clusters))
        df['count'] = [list(clusters).count(i) for i in sorted(set(clusters))]
        df['percentage'] = df['count'] / df['count'].sum() * 100
        df = df.reset_index(drop=True)
        return df

    def get_scores(self, clusters):
        """
        Method to calculate the silhouette score and Davies-Bouldin score.
        :param clusters: list - The cluster labels.
        :return: float - The silhouette score.
        :return: float - The Davies-Bouldin score.
        """
        scores = pd.DataFrame()
        sil = silhouette_score(self.df, clusters)
        dav = davies_bouldin_score(self.df, clusters)
        scores['silhouette_score'] = [sil]
        scores['davies_bouldin_score'] = [dav]
        return scores
