from pandas import DataFrame
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
from sklearn.cluster import AgglomerativeClustering, DBSCAN, Birch
from sklearn.mixture import GaussianMixture
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px


class UVyper:
    def __init__(self, data: str):
        """
        Method to read and initialize the data.
        :param data: str - path to the preprocessed data
        """
        self.df = pd.read_csv(data)
        self.score_table = pd.DataFrame()
        self.distribution = pd.DataFrame()

    def kmeans_w(self, minK: int, maxK: int, metric: str, min_size_per: float, max_size_per: float,
                 rand_sample_prop: float, filename: str, dataset: str,
                 n_clusters: int = None):
        """
        Method to find the clusters using KMeans.
        :param minK: int - The minimum number of clusters to consider.
        :param maxK: int - The maximum number of clusters to consider.
        :param metric: str - optional (default='distortion'). The metric used to quantify the
        quality of clustering. Possible options include distortion, silhouette, calinski_harabasz,
        davies_bouldin, and others.
        :param min_size_per: float - percentage of minimum size of cluster
        :param max_size_per: float - percentage of maximum size of cluster
        :param rand_sample_prop: float - random sampling proportion
        :param filename:  float - path of the pickle file
        :param dataset: float - path of the original dataset
        :param n_clusters: int - no of clusters
        :return:  labels, summary table, scatter plots
        """

        def elbow(minK: int, maxK: int, metric: str = 'distortion'):
            """
            Method to identify the optimal number of clusters (k) for a given dataset using the elbow method.
            :param minK: int - The minimum number of clusters to consider.
            :param maxK: int - The maximum number of clusters
            to consider.
            :param metric: str - optional (default='distortion'). The metric used to quantify the
            quality of clustering. Possible options include distortion, silhouette, calinski_harabasz,
            davies_bouldin, and others.
            :return: int - The optimal number of clusters (k).
            """
            X = self.df
            model = KMeans()
            visualizer = KElbowVisualizer(model, k=(minK, maxK), metric=metric, timings=False)
            visualizer.fit(X)
            visualizer.show()
            return visualizer.elbow_value_

        def kmeans(n_clusters: int, min_size_per: float, max_size_per: float):
            """
            Method to find the clusters using KMeans.
            :param n_clusters: no of clusters
            :param min_size_per: percentage of minimum size of cluster
            :param max_size_per: percentage of maximum size of cluster
            :return: ndarray - The cluster labels.
            """
            kmeanModel = KMeansConstrained(
                n_clusters=n_clusters,
                size_min=int((min_size_per / 100) * self.df.shape[0]),
                size_max=int((max_size_per / 100) * self.df.shape[0]), )
            clusters = kmeanModel.fit_predict(self.df)
            return clusters

        def kmeans_model_create(n_clusters: int, min_size_per: float, max_size_per: float, filename: str,
                                rand_sample_prop: float = 1.0):
            """
            Method to create the KMeans model as a pickle file.
            :param filename: name of the pickle file
            :param n_clusters: no of clusters
            :param min_size_per: percentage of minimum size of cluster
            :param max_size_per: percentage of maximum size of cluster
            :param rand_sample_prop: random sampling proportion
            """
            sample_data = self.df.sample(frac=rand_sample_prop, random_state=42)
            kmeanModel = KMeansConstrained(
                n_clusters=n_clusters,
                size_min=int((min_size_per / 100) * sample_data.shape[0]),
                size_max=int((max_size_per / 100) * sample_data.shape[0]),
            )
            kmeanModel.fit(sample_data)
            with open(filename, 'wb') as file:
                pickle.dump(kmeanModel, file)

        def kmeans_model_read(filename: str):
            """
            Method to read the KMeans model from a pickle file.
            :param filename: str - The path of the pickle file.
            :return: ndarray - The cluster labels.
            """
            with open(filename, 'rb') as file:
                kmeanModel = pickle.load(file)
            clusters = kmeanModel.predict(self.df)
            return clusters

        def pca(clusters: np.ndarray, n_components: int = 2):
            """
            Method to perform PCA.
            :param clusters: ndarray - The cluster labels.
            :param n_components: int - The number of components to consider.
            :return: dataframe - The PCA results.
            """
            pca = PCA(n_components=n_components)
            principalComponents = pca.fit_transform(self.df)
            principalDf = pd.DataFrame(data=principalComponents,
                                       columns=['PC' + str(i) for i in range(1, n_components + 1)])
            principalDf['cluster'] = clusters
            return principalDf

        def scatter_plot_2d(principalComponents: pd.DataFrame):
            """
            Method to plot the PCA results.
            :param principalComponents: dataframe - The PCA results.
            :return: Scatter Plot
            """
            plt.scatter(principalComponents['PC1'], principalComponents['PC2'], c=principalComponents['cluster'],
                        cmap=viridis)
            plt.title('K-Means Clustering')
            plt.colorbar()
            plt.show()

        def scatter_plot_3d(principalComponents: pd.DataFrame):
            """
            Method to plot the PCA results.
            :param principalComponents: dataframe - The PCA results.
            :return: Scatter Plot
            """
            fig = px.scatter_3d(principalComponents, x='PC1', y='PC2', z='PC3', color='cluster', title='K-Means '
                                                                                                       'Clustering')
            fig.show()

        def get_cluster_centers(clusters: np.ndarray, dataset: str):
            """
            Method to calculate the cluster centers.
            :param dataset: str - path to the dataset
            :param clusters: list - The cluster labels.
            :return: DataFrame - The cluster centers.
            """
            temp = pd.read_csv(dataset)
            temp['cluster'] = clusters
            return temp.groupby(clusters).mean()

        def get_cluster_distribution(clusters: np.ndarray):
            """
            Method to calculate the distribution of clusters.
            :param clusters: ndarray - The cluster labels.
            :return: dataframe - The distribution of clusters.
            """
            df = pd.DataFrame()
            df['cluster'] = sorted(set(clusters))
            df['count'] = [list(clusters).count(i) for i in sorted(set(clusters))]
            df['percentage'] = df['count'] / df['count'].sum() * 100
            df['Model'] = ['Kmeans' for i in range(len(set(clusters)))]
            df = df.reset_index(drop=True)
            return df

        def get_scores(clusters: np.ndarray):
            """
            Method to calculate the silhouette score and Davies-Bouldin score.
            :param clusters: list - The cluster labels.
            :return: float - The silhouette score.
            :return: float - The Davies-Bouldin score.
            """
            scores = pd.DataFrame()
            sil = silhouette_score(self.df, clusters)
            dav = davies_bouldin_score(self.df, clusters)
            scores['Model'] = ['KMeans']
            scores['Silhouette'] = [sil]
            scores['Davies Bouldin'] = [dav]
            scores['n_clusters'] = [len(set(clusters))]
            return scores

        print("Performing KMeans Clustering...")
        if n_clusters is None:
            n_clusters = elbow(minK, maxK, metric)
        option = int(input("Do you want to save the model? (1/0)"))
        if option == 1:
            kmeans_model_create(n_clusters=n_clusters, min_size_per=min_size_per, max_size_per=max_size_per,
                                filename=filename, rand_sample_prop=rand_sample_prop)
            clusters = kmeans_model_read(filename)
        else:
            clusters = kmeans(n_clusters=n_clusters, min_size_per=min_size_per, max_size_per=max_size_per)
        principalComponents_2d = pca(clusters, n_components=2)
        principalComponents_3d = pca(clusters, n_components=3)
        scatter_plot_2d(principalComponents_2d)
        scatter_plot_3d(principalComponents_3d)
        cluster_centers = get_cluster_centers(clusters, dataset)
        cluster_distribution = get_cluster_distribution(clusters)
        scores = get_scores(clusters)
        self.score_table = self.score_table.append(scores, ignore_index=True)
        self.distribution = self.distribution.append(cluster_distribution, ignore_index=True)
        print(cluster_distribution)
        print(scores)
        print(cluster_centers)
        print("KMeans Clustering Complete!")

    def hierarchical_w(self, param_grid: dict, folds: int, n_iter: int, rand_sample_prop: float, dataset: str,
                       n_clusters: int = None, linkage: str = None,
                       affinity: str = None):
        """
        Method to perform hierarchical clustering.
        :param param_grid: dict - parameters grid
        :param folds: int - number of folds
        :param n_iter: int - number of iterations
        :param rand_sample_prop: float - random sample proportion
        :param dataset: str - path to the original dataset
        :param n_clusters: int - number of clusters
        :param linkage: str - linkage
        :param affinity: str - affinity
        :return: labels, summary table, scatter plots
        """

        def silhouettee(estimator: object, df: pd.DataFrame, metric: str = 'euclidean'):
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

        def randomizedSearchCV_hierarchical(grid: dict, cv: int = 5, n_iter: int = 10, rand_sample_prop: float = 0.2):
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
                                               scoring=silhouettee, n_iter=n_iter)
            random_search.fit(sample_data)
            training_history = random_search.cv_results_
            return random_search.best_score_, random_search.best_params_, training_history

        def hierarchical(n_clusters: int, linkage: str, affinity: str, random_sample_prop: float = 1.0):
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

        def knn(sample_data: pd.DataFrame, clusters: np.ndarray, n_neighbors: int = 5):
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

        def pca(clusters: np.ndarray, n_components: int):
            """
            Method to perform PCA.
            :param clusters: ndarray - The cluster labels.
            :param n_components: int - The number of components to consider.
            :return: dataframe - The PCA results.
            """
            pca = PCA(n_components=n_components)
            principalComponents = pca.fit_transform(self.df)
            principalDf = pd.DataFrame(data=principalComponents,
                                       columns=['PC' + str(i) for i in range(1, n_components + 1)])
            principalDf['cluster'] = clusters
            return principalDf

        def scatter_plot_2d(principalComponents: pd.DataFrame):
            """
            Method to plot the PCA results.
            :param principalComponents: dataframe - The PCA results.
            :return: Scatter Plot
            """
            plt.scatter(principalComponents['PC1'], principalComponents['PC2'], c=principalComponents['cluster'],
                        cmap=viridis)
            plt.title('Hierarchical Clustering')
            plt.colorbar()
            plt.show()

        def scatter_plot_3d(principalComponents: pd.DataFrame):
            """
            Method to plot the PCA results.
            :param principalComponents: dataframe - The PCA results.
            :return: Scatter Plot
            """
            fig = px.scatter_3d(principalComponents, x='PC1', y='PC2', z='PC3', color='cluster',
                                title='Hierarchical Clustering')
            fig.show()

        # @staticmethod
        def get_cluster_centers(clusters: np.ndarray, dataset: str):
            """
            Method to calculate the cluster centers.
            :param dataset: str - path to the dataset
            :param clusters: list - The cluster labels.
            :return: DataFrame - The cluster centers.
            """
            temp = pd.read_csv(dataset)
            temp['cluster'] = clusters
            return temp.groupby(clusters).mean()

        def get_cluster_distribution(clusters: np.ndarray):
            """
            Method to calculate the distribution of clusters.
            :param clusters: ndarray - The cluster labels.
            :return: dataframe - The distribution of clusters.
            """
            df = pd.DataFrame()
            df['cluster'] = sorted(set(clusters))
            df['count'] = [list(clusters).count(i) for i in sorted(set(clusters))]
            df['percentage'] = df['count'] / df['count'].sum() * 100
            df['Model'] = ['Hierarchical' for i in range(len(set(clusters)))]
            df = df.reset_index(drop=True)
            return df

        def get_scores(clusters: np.ndarray):
            """
            Method to calculate the silhouette score and Davies-Bouldin score.
            :param clusters: list - The cluster labels.
            :return: float - The silhouette score.
            :return: float - The Davies-Bouldin score.
            """
            scores = pd.DataFrame()
            sil = silhouette_score(self.df, clusters)
            dav = davies_bouldin_score(self.df, clusters)
            scores['Model'] = ['Hierarchical']
            scores['Silhouette'] = [sil]
            scores['Davies Bouldin'] = [dav]
            scores['n_clusters'] = [len(set(clusters))]
            return scores

        print("Performing Hierarchical Clustering...")
        if n_clusters is None or linkage is None or affinity is None:
            a, b, c = randomizedSearchCV_hierarchical(grid=param_grid, cv=folds, n_iter=n_iter,
                                                      rand_sample_prop=rand_sample_prop)
            if n_clusters is None:
                n_clusters = b['n_clusters']
                print("Recommended number of clusters: ", n_clusters)
            if linkage is None:
                linkage = b['linkage']
                print("Recommended linkage: ", linkage)
            if affinity is None:
                affinity = b['affinity']
                print("Recommended affinity: ", affinity)

        clusters, sample_data = hierarchical(n_clusters=n_clusters, linkage=linkage, affinity=affinity,
                                             random_sample_prop=rand_sample_prop)
        clusters = knn(sample_data, clusters)
        principalComponents_2d = pca(clusters, n_components=2)
        principalComponents_3d = pca(clusters, n_components=3)
        scatter_plot_2d(principalComponents_2d)
        scatter_plot_3d(principalComponents_3d)
        cluster_centers = get_cluster_centers(clusters, dataset)
        cluster_distribution = get_cluster_distribution(clusters)
        scores = get_scores(clusters)
        self.distribution = self.distribution.append(cluster_distribution, ignore_index=True)
        self.score_table = self.score_table.append(scores, ignore_index=True)
        print(cluster_distribution)
        print(scores)
        print(cluster_centers)
        print("Hierarchical Clustering Complete!")

    def gmm_w(self, param_grid: dict, folds: int, n_iter: int, rand_sample_prop: float, filename: str, dataset: str,
              n_components: int = None,
              covariance_type: str = None,
              init_params: str = None):
        """
        Method to perform Gaussian Mixture Model clustering.
        :param param_grid: dict - The parameters to be used for the randomized search cross validation.
        :param folds: int - The number of folds to be used for the cross validation.
        :param n_iter: int - The number of iterations to be used for the randomized search cross validation.
        :param rand_sample_prop: float - random sample proportion
        :param filename: str - path to the pickle file
        :param dataset: str - path to the original dataset
        :param n_components: int - number of components
        :param covariance_type: str - covariance type
        :param init_params: str - initialization parameters
        :return: labels, summary table, scatter plots
        """

        def silhouettee(estimator: object, df: pd.DataFrame, metric: str = 'euclidean'):
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

        def randomizedSearchCV_gmm(grid: dict, cv: int = 5, n_iter: int = 10, rand_sample_prop: float = 0.2):
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
            random_search = RandomizedSearchCV(estimator=gmm, param_distributions=grid, cv=cv, scoring=silhouettee,
                                               n_iter=n_iter)
            random_search.fit(sample_data)
            return random_search.best_score_, random_search.best_params_, random_search.cv_results_

        def gmm(n_components: int, covariance_type: str, init_params: str):
            """
            Method to perform GMM clustering.
            :param n_components: int - number of clusters
            :param covariance_type: str - covariance type
            :param init_params: str - initialization method
            :return: numpy array - cluster labels
            """
            gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, init_params=init_params)
            clusters = gmm.fit_predict(self.df)
            return clusters

        def gmm_model_create(n_components: int, covariance_type: str, init_params: str, filename: str,
                             rand_sample_prop: float = 1.0):
            """
            Method to create pickle file of gmm model.
            :param filename: name of the pickle file
            :param n_components: int - number of clusters
            :param covariance_type: str - covariance type
            :param init_params: str - initialization method
            :param rand_sample_prop: float - proportion of data to sample
            :return: numpy array - cluster labels
            """
            sample_data = self.df.sample(frac=rand_sample_prop, random_state=42)
            gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, init_params=init_params)
            gmm.fit(sample_data)
            with open(filename, 'wb') as f:
                pickle.dump(gmm, f)

        def gmm_model_read(filename: str):
            """
            Method to load the pickle file of gmm model.
            :param filename: str - path to the pickle file
            :return: ndarray - cluster labels
            """
            with open(filename, 'rb') as f:
                gmm = pickle.load(f)
            clusters = gmm.predict(self.df)
            return clusters

        def pca(clusters: np.ndarray, n_components: int):
            """
            Method to perform PCA.
            :param clusters: ndarray - The cluster labels.
            :param n_components: int - The number of components to consider.
            :return: dataframe - The PCA results.
            """
            pca = PCA(n_components=n_components)
            principalComponents = pca.fit_transform(self.df)
            principalDf = pd.DataFrame(data=principalComponents,
                                       columns=['PC' + str(i) for i in range(1, n_components + 1)])
            principalDf['cluster'] = clusters
            return principalDf

        def scatter_plot_2d(principalComponents: pd.DataFrame):
            """
            Method to plot the PCA results.
            :param principalComponents: dataframe - The PCA results.
            :return: Scatter Plot
            """
            plt.scatter(principalComponents['PC1'], principalComponents['PC2'], c=principalComponents['cluster'],
                        cmap=viridis)
            plt.title('GMM')
            plt.colorbar()
            plt.show()

        def scatter_plot_3d(principalComponents: pd.DataFrame):
            """
            Method to plot the PCA results.
            :param principalComponents: dataframe - The PCA results.
            :return: Scatter Plot
            """
            fig = px.scatter_3d(principalComponents, x='PC1', y='PC2', z='PC3', color='cluster',
                                title='GMM')
            fig.show()

        # @staticmethod
        def get_cluster_centers(clusters: np.ndarray, dataset: str):
            """
            Method to calculate the cluster centers.
            :param dataset: str - path to the dataset
            :param clusters: list - The cluster labels.
            :return: DataFrame - The cluster centers.
            """
            temp = pd.read_csv(dataset)
            temp['cluster'] = clusters
            return temp.groupby(clusters).mean()

        def get_cluster_distribution(clusters: np.ndarray):
            """
            Method to calculate the distribution of clusters.
            :param clusters: ndarray - The cluster labels.
            :return: dataframe - The distribution of clusters.
            """
            df = pd.DataFrame()
            df['cluster'] = sorted(set(clusters))
            df['count'] = [list(clusters).count(i) for i in sorted(set(clusters))]
            df['percentage'] = df['count'] / df['count'].sum() * 100
            df['Model'] = ['GMM' for i in range(len(set(clusters)))]
            df = df.reset_index(drop=True)
            return df

        def get_scores(clusters: np.ndarray):
            """
            Method to calculate the silhouette score and Davies-Bouldin score.
            :param clusters: list - The cluster labels.
            :return: float - The silhouette score.
            :return: float - The Davies-Bouldin score.
            """
            scores = pd.DataFrame()
            sil = silhouette_score(self.df, clusters)
            dav = davies_bouldin_score(self.df, clusters)
            scores['Model'] = ['GMM']
            scores['Silhouette'] = [sil]
            scores['Davies Bouldin'] = [dav]
            scores['n_clusters'] = [len(set(clusters))]
            return scores

        print("Performing GMM Clustering...")
        if n_components is None or covariance_type is None or init_params is None:
            a, b, c = randomizedSearchCV_gmm(grid=param_grid, cv=folds, n_iter=n_iter,
                                             rand_sample_prop=rand_sample_prop)
            if n_components is None:
                n_components = b['n_components']
                print("Recommended number of clusters: ", n_components)
            if covariance_type is None:
                covariance_type = b['covariance_type']
                print("Recommended covariance type: ", covariance_type)
            if init_params is None:
                init_params = b['init_params']
                print("Recommended initialization method: ", init_params)
        option = int(input("Do you want to save the model? (1/0): "))
        if option == 1:
            gmm_model_create(n_components=n_components, covariance_type=covariance_type, init_params=init_params,
                             filename=filename, rand_sample_prop=rand_sample_prop)
            clusters = gmm_model_read(filename=filename)
        else:
            clusters = gmm(n_components=n_components, covariance_type=covariance_type, init_params=init_params)
        principalComponents_2d = pca(clusters, n_components=2)
        principalComponents_3d = pca(clusters, n_components=3)
        scatter_plot_2d(principalComponents_2d)
        scatter_plot_3d(principalComponents_3d)
        cluster_centers = get_cluster_centers(clusters, dataset)
        cluster_distribution = get_cluster_distribution(clusters)
        scores = get_scores(clusters)
        self.distribution = self.distribution.append(cluster_distribution, ignore_index=True)
        self.score_table = self.score_table.append(scores, ignore_index=True)
        print(cluster_distribution)
        print(scores)
        print(cluster_centers)
        print("GMM Clustering Complete!")

    def birch_w(self, param_grid: dict, folds: int, n_iter: int, rand_sample_prop: float, filename: str, dataset: str,
                n_clusters: int = None,
                branching_factor: int = None,
                threshold: float = None):
        """
        Method to perform Birch clustering.
        :param param_grid: dict - The parameters to be used for the randomized search cross validation.
        :param folds: int - The number of folds to be used for the cross validation.
        :param n_iter: int - The number of iterations to be used for the randomized search cross validation.
        :param rand_sample_prop: float - random sample proportion
        :param filename: str - path to the pickle file
        :param dataset: str - path to the original dataset
        :param n_clusters: int - number of clusters
        :param branching_factor: int - branching factor
        :param threshold: int - threshold
        :return: labels, summary table, scatter plots
        """

        def silhouettee(estimator: object, df: pd.DataFrame, metric: str = 'euclidean'):
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

        def randomizedSearchCV_birch(grid: dict, cv: int = 5, n_iter: int = 10, rand_sample_prop: float = 0.2):
            """
            Method to perform randomized search cross validation on the GMM algorithm.
            :param grid: dict - dictionary of parameters to search over
            :param cv: int - number of folds
            :param n_iter: int - number of iterations
            :param rand_sample_prop: float - proportion of data to sample
            :return: tuple - best score, the best parameters, and cross validation results
            """
            sample_data = self.df.sample(frac=rand_sample_prop)
            birch = Birch()
            random_search = RandomizedSearchCV(estimator=birch, param_distributions=grid, cv=cv, scoring=silhouettee,
                                               n_iter=n_iter)
            random_search.fit(sample_data)
            return random_search.best_score_, random_search.best_params_, random_search.cv_results_

        def birch(n_clusters: int, branching_factor: int, threshold: float):
            """
            Method to perform GMM clustering.
            :param n_clusters: int - number of clusters
            :param branching_factor: int - branching factor
            :param threshold: int - threshold
            :return: numpy array - cluster labels
            """
            birch = Birch(n_clusters=n_clusters, branching_factor=branching_factor, threshold=threshold)
            clusters = birch.fit_predict(self.df)
            return clusters

        def birch_model_create(n_clusters: int, branching_factor: int, threshold: float, filename: str,
                               rand_sample_prop: float = 1.0):
            """
            Method to create pickle file of gmm model.
            :param filename: str - path to the pickle file
            :param n_clusters: int - number of clusters
            :param branching_factor: int - branching factor
            :param threshold: int - threshold
            :param rand_sample_prop: float - proportion of data to sample
            :return: numpy array - cluster labels
            """
            sample_data = self.df.sample(frac=rand_sample_prop, random_state=42)
            birch = Birch(n_clusters=n_clusters, branching_factor=branching_factor, threshold=threshold)
            birch.fit(sample_data)
            with open(filename, 'wb') as f:
                pickle.dump(birch, f)

        def birch_model_read(filename: str):
            """
            Method to load the pickle file of gmm model.
            :param filename: str - path to the pickle file
            :return: ndarray - cluster labels
            """
            with open(filename, 'rb') as f:
                birch = pickle.load(f)
            clusters = birch.predict(self.df)
            return clusters

        def pca(clusters: np.ndarray, n_components: int):
            """
            Method to perform PCA.
            :param clusters: ndarray - The cluster labels.
            :param n_components: int - The number of components to consider.
            :return: dataframe - The PCA results.
            """
            pca = PCA(n_components=n_components)
            principalComponents = pca.fit_transform(self.df)
            principalDf = pd.DataFrame(data=principalComponents,
                                       columns=['PC' + str(i) for i in range(1, n_components + 1)])
            principalDf['cluster'] = clusters
            return principalDf

        def scatter_plot_2d(principalComponents: pd.DataFrame):
            """
            Method to plot the PCA results.
            :param principalComponents: dataframe - The PCA results.
            :return: Scatter Plot
            """
            plt.scatter(principalComponents['PC1'], principalComponents['PC2'], c=principalComponents['cluster'],
                        cmap=viridis)
            plt.title('Birch Clustering')
            plt.colorbar()
            plt.show()

        def scatter_plot_3d(principalComponents: pd.DataFrame):
            """
            Method to plot the PCA results.
            :param principalComponents: dataframe - The PCA results.
            :return: Scatter Plot
            """
            fig = px.scatter_3d(principalComponents, x='PC1', y='PC2', z='PC3', color='cluster',
                                title='Birch Clustering')
            fig.show()

        # @staticmethod
        def get_cluster_centers(clusters: np.ndarray, dataset: str):
            """
            Method to calculate the cluster centers.
            :param dataset: str - path to the dataset
            :param clusters: list - The cluster labels.
            :return: DataFrame - The cluster centers.
            """
            temp = pd.read_csv(dataset)
            temp['cluster'] = clusters
            return temp.groupby(clusters).mean()

        def get_cluster_distribution(clusters: np.ndarray):
            """
            Method to calculate the distribution of clusters.
            :param clusters: ndarray - The cluster labels.
            :return: dataframe - The distribution of clusters.
            """
            df = pd.DataFrame()
            df['cluster'] = sorted(set(clusters))
            df['count'] = [list(clusters).count(i) for i in sorted(set(clusters))]
            df['percentage'] = df['count'] / df['count'].sum() * 100
            df['Model'] = ['Birch' for i in range(len(set(clusters)))]
            df = df.reset_index(drop=True)
            return df

        def get_scores(clusters: np.ndarray):
            """
            Method to calculate the silhouette score and Davies-Bouldin score.
            :param clusters: list - The cluster labels.
            :return: float - The silhouette score.
            :return: float - The Davies-Bouldin score.
            """
            scores = pd.DataFrame()
            sil = silhouette_score(self.df, clusters)
            dav = davies_bouldin_score(self.df, clusters)
            scores['Model'] = ['Birch']
            scores['Silhouette'] = [sil]
            scores['Davies Bouldin'] = [dav]
            scores['n_clusters'] = [len(set(clusters))]
            return scores

        print("Performing Birch Clustering...")
        if n_clusters is None or branching_factor is None or threshold is None:
            a, b, c = randomizedSearchCV_birch(grid=param_grid, cv=folds, n_iter=n_iter,
                                               rand_sample_prop=rand_sample_prop)
            if n_clusters is None:
                n_clusters = b['n_clusters']
                print("Recommended number of clusters: ", n_clusters)
            if branching_factor is None:
                branching_factor = b['branching_factor']
                print("Recommended branching factor: ", branching_factor)
            if threshold is None:
                threshold = b['threshold']
                print("Recommended threshold: ", threshold)
        option = int(input("Do you want to save the model? (1/0): "))
        if option == 1:
            birch_model_create(n_clusters=n_clusters, branching_factor=branching_factor, threshold=threshold,
                               filename=filename, rand_sample_prop=rand_sample_prop)
            clusters = birch_model_read(filename=filename)
        else:
            clusters = birch(n_clusters=n_clusters, branching_factor=branching_factor, threshold=threshold)
        principalComponents_2d = pca(clusters, n_components=2)
        principalComponents_3d = pca(clusters, n_components=3)
        scatter_plot_2d(principalComponents_2d)
        scatter_plot_3d(principalComponents_3d)
        cluster_centers = get_cluster_centers(clusters, dataset)
        cluster_distribution = get_cluster_distribution(clusters)
        scores = get_scores(clusters)
        self.distribution = self.distribution.append(cluster_distribution, ignore_index=True)
        self.score_table = self.score_table.append(scores, ignore_index=True)
        print(cluster_distribution)
        print(scores)
        print(cluster_centers)
        print("Birch Clustering Complete!")

    def get_models_summary(self):
        """
        Method to get the summary of the clustering
        :return: dataframe - summary of the clustering
        """

        def sort_score_table(score_df: pd.DataFrame):
            silhouette_ranks = score_df['Silhouette'].rank(method='dense', ascending=False)
            db_ranks = score_df['Davies Bouldin'].rank(method='dense')
            combined_ranks = (silhouette_ranks + db_ranks) / 2
            score_df['Rank'] = combined_ranks.rank(method='dense')
            score_df = score_df.sort_values(by=['Rank'])
            return score_df

        def get_distribution_graph(distribution_table: pd.DataFrame):
            df = distribution_table.pivot(index='Model', columns='cluster', values='percentage')
            df.reset_index(inplace=True)
            ax = df.plot(x='Model', kind='bar', stacked=True, figsize=(10, 5), title='Cluster Distribution')
            plt.xticks(rotation=0)
            for p in ax.containers:
                ax.bar_label(p, label_type='center', labels=[f'{val:.2f}%' if val > 0 else '' for val in p.datavalues],
                             fontsize=10)
            plt.show()

        ranked_score_table = sort_score_table(self.score_table)
        get_distribution_graph(self.distribution)
        print(ranked_score_table)
