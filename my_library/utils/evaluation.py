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


def sort_score_table(score_df):
    silhouette_ranks = score_df['Silhouette'].rank(method='dense', ascending=False)
    db_ranks = score_df['Davies Bouldin'].rank(method='dense')
    combined_ranks = (silhouette_ranks + db_ranks) / 2
    score_df['Rank'] = combined_ranks.rank(method='dense')
    score_df = score_df.sort_values(by=['Rank'])
    return score_df


def get_score_table(kmeans_score=None, hierarchical_score=None, gmm_score=None, birch_score=None, dbscan_score=None):
    score_df = pd.DataFrame()
    if kmeans_score is not None:
        score_df = score_df.append(kmeans_score)
    if hierarchical_score is not None:
        score_df = score_df.append(hierarchical_score)
    if gmm_score is not None:
        score_df = score_df.append(gmm_score)
    if birch_score is not None:
        score_df = score_df.append(birch_score)
    if dbscan_score is not None:
        score_df = score_df.append(dbscan_score)
    score_df.reset_index(inplace=True, drop=True)
    return score_df


def get_distribution_table(kmeans_distribution=None, hierarchical_distribution=None, gmm_distribution=None,
                           birch_distribution=None, dbscan_distribution=None):
    distribution_df = pd.DataFrame()
    if kmeans_distribution is not None:
        distribution_df = distribution_df.append(kmeans_distribution)
    if hierarchical_distribution is not None:
        distribution_df = distribution_df.append(hierarchical_distribution)
    if gmm_distribution is not None:
        distribution_df = distribution_df.append(gmm_distribution)
    if birch_distribution is not None:
        distribution_df = distribution_df.append(birch_distribution)
    if dbscan_distribution is not None:
        distribution_df = distribution_df.append(dbscan_distribution)
    distribution_df.reset_index(inplace=True, drop=True)
    return distribution_df


def get_distribution_graph(distribution_table):
    df = distribution_table.pivot(index='Model', columns='cluster', values='percentage')
    df.reset_index(inplace=True)
    ax = df.plot(x='Model', kind='bar', stacked=True, figsize=(10, 5), title='Cluster Distribution')
    plt.xticks(rotation=0)
    for p in ax.containers:
        ax.bar_label(p, label_type='center', labels=[f'{val:.2f}%' if val > 0 else '' for val in p.datavalues],
                     fontsize=10)
    plt.show()


class evaluation:
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
                    sum(np.min(cdist(self.df, kmeanModel_minSize.cluster_centers_, 'euclidean'), axis=1)) /
                    self.df.shape[
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
                    sum(np.min(cdist(self.df, kmeanModel_maxSize.cluster_centers_, 'euclidean'), axis=1)) /
                    self.df.shape[
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

        def kmeans(self, n_clusters, min_size_per, max_size_per, rand_samp_prop=1.0):
            """
            Method to find the clusters using KMeans.
            :param n_clusters: no of clusters
            :param min_size_per: percentage of minimum size of cluster
            :param max_size_per: percentage of maximum size of cluster
            :param rand_samp_prop: random sampling proportion
            :return: ndarray - The cluster labels.
            """
            sample_data = self.df.sample(frac=rand_samp_prop, random_state=42)
            kmeanModel = KMeansConstrained(
                n_clusters=n_clusters,
                size_min=int((min_size_per / 100) * sample_data.shape[0]),
                size_max=int((max_size_per / 100) * sample_data.shape[0]), )
            kmeanModel.fit(sample_data)
            with open('kmeanModel.pkl', 'wb') as file:
                pickle.dump(kmeanModel, file)
            with open('kmeanModel.pkl', 'rb') as file:
                kmeanModel = pickle.load(file)
            clusters = kmeanModel.predict(self.df)
            return clusters

        def kmeans_model_create(self, n_clusters, min_size_per, max_size_per, rand_samp_prop=1.0):
            """
            Method to create the KMeans model as a pickle file.
            :param n_clusters: no of clusters
            :param min_size_per: percentage of minimum size of cluster
            :param max_size_per: percentage of maximum size of cluster
            :param rand_samp_prop: random sampling proportion
            """
            sample_data = self.df.sample(frac=rand_samp_prop, random_state=42)
            kmeanModel = KMeansConstrained(
                n_clusters=n_clusters,
                size_min=int((min_size_per / 100) * sample_data.shape[0]),
                size_max=int((max_size_per / 100) * sample_data.shape[0]), )
            kmeanModel.fit(sample_data)
            with open('kmeanModel.pkl', 'wb') as file:
                pickle.dump(kmeanModel, file)

        def kmeans_model_read(self, filename):
            """
            Method to read the KMeans model from a pickle file.
            :param filename: str - The path of the pickle file.
            :return: ndarray - The cluster labels.
            """
            with open(filename, 'rb') as file:
                kmeanModel = pickle.load(file)
            clusters = kmeanModel.predict(self.df)
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
            df['Model'] = ['Kmeans' for i in range(len(set(clusters)))]
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
            scores['Model'] = ['KMeans']
            scores['Silhouette'] = [sil]
            scores['Davies Bouldin'] = [dav]
            scores['n_clusters'] = [len(set(clusters))]
            return scores

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
            df['Model'] = ['Hierarchical' for i in range(len(set(clusters)))]
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
            scores['Model'] = ['Hierarchical']
            scores['Silhouette'] = [sil]
            scores['Davies Bouldin'] = [dav]
            scores['n_clusters'] = [len(set(clusters))]
            return scores

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
            df['Model'] = ['GMM' for i in range(len(set(clusters)))]
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
            scores['Model'] = ['GMM']
            scores['Silhouette'] = [sil]
            scores['Davies Bouldin'] = [dav]
            scores['n_clusters'] = [len(set(clusters))]
            return scores

    class birch:
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
            random_search = RandomizedSearchCV(estimator=birch, param_distributions=grid, cv=cv,
                                               scoring=self.silhouette,
                                               n_iter=n_iter)
            random_search.fit(sample_data)
            return random_search.best_score_, random_search.best_params_, random_search.cv_results_

        def birch(self, branching_factor: int, n_clusters: int, threshold: float, rand_sample_prop: float = 1.0):
            """
            Method to perform birch clustering
            :param branching_factor: int - branching factor
            :param n_clusters: int - number of clusters
            :param threshold: float - threshold
            :param rand_sample_prop: float - proportion of data to sample, default is 1.0
            :return: ndarray - cluster labels
            """
            sample_data = self.df.sample(frac=rand_sample_prop, random_state=42)
            birch = Birch(branching_factor=branching_factor, n_clusters=n_clusters, threshold=threshold)
            birch.fit(sample_data)
            with open('birchModel.pkl', 'wb') as f:
                pickle.dump(birch, f)
            with open('birchModel.pkl', 'rb') as f:
                birch = pickle.load(f)
            clusters = birch.predict(self.df)
            return clusters

        def birch_model_create(self, branching_factor: int, n_clusters: int, threshold: float,
                               rand_sample_prop: float = 1.0):
            """
            Method to create a pickle file of the birch model
            :param branching_factor: int - branching factor
            :param n_clusters: int - number of clusters
            :param threshold: float - threshold
            :param rand_sample_prop: float - proportion of data to sample, default is 1.0
            :return: ndarray - cluster labels
            """
            sample_data = self.df.sample(frac=rand_sample_prop, random_state=42)
            birch = Birch(branching_factor=branching_factor, n_clusters=n_clusters, threshold=threshold)
            birch.fit(sample_data)
            with open('birchModel.pkl', 'wb') as f:
                pickle.dump(birch, f)

        def birch_model_read(self, filename):
            """
            Method to read the pickle file of the birch model
            :param filename: str - path of the pickle file
            :return: ndarray - cluster labels
            """
            with open(filename, 'rb') as f:
                birch = pickle.load(f)
            clusters = birch.predict(self.df)
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
            df['Model'] = ['Birch' for i in range(len(set(clusters)))]
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
            scores['Model'] = ['Birch']
            scores['Silhouette'] = [sil]
            scores['Davies Bouldin'] = [dav]
            scores['n_clusters'] = [len(set(clusters))]
            return scores
