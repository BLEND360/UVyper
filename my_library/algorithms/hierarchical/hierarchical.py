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
from sklearn.neighbors import KNeighborsClassifier


class Hierarchical:
    def __init__(self, data):
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
    def silhouette(estimator, df, metric='euclidean'):
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
    def get_training_history(training_history):
        """
        Method to get the training history
        :param training_history: dict - training history
        :return:
        """
        return pd.DataFrame.from_dict(training_history)

    def gridSearchCV_hierarchical(self, grid, cv=5):
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

    def randomizedSearchCV_hierarchical(self, grid, cv=5, n_iter=10, rand_sample_prop=0.2):
        """
        Method to perform randomized search cross validation
        :param grid: dict - grid
        :param cv: int - number of folds
        :param n_iter: int - number of iterations
        :param rand_sample_prop: float - random sample proportion
        :return: tuple - best score, best parameters, training history
        """
        sample_data = self.df.sample(frac=rand_sample_prop)
        hierarchical = AgglomerativeClustering()
        random_search = RandomizedSearchCV(estimator=hierarchical, param_distributions=grid, cv=cv,
                                           scoring=self.silhouette, n_iter=n_iter)
        random_search.fit(sample_data)
        training_history = random_search.cv_results_
        return random_search.best_score_, random_search.best_params_, training_history

    def hierarchical(self, n_clusters, linkage, affinity, random_sample_prop=1.0):
        """
        Method to perform hierarchical clustering
        :param n_clusters: int - number of clusters
        :param linkage: str - linkage
        :param affinity: str - affinity
        :param random_sample_prop: float - random sample proportion
        :return: ndarray - clusters
        """
        sample_data = self.df.sample(frac=random_sample_prop, random_state=42)
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, linkage=linkage)
        clusters = hierarchical.fit_predict(sample_data)
        return clusters, sample_data

    def knn(self, sample_data, clusters, n_neighbors=5):
        """
        Method to perform knn
        :param sample_data: dataframe - sample data used in hierarchical method
        :param clusters: ndarray - clusters from hierarchical method
        :param n_neighbors: int - number of neighbors
        :return: ndarray - clusters labels for the entire dataset
        """

        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(sample_data, clusters)
        clu = knn.predict(self.df)
        return clu

    def pca(self, clusters, n_components=2):
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
