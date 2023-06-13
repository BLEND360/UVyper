# Customer Segmentation Tool

## Introduction

This tool is designed to help you segment your customers based on the features available in dataset. It will help you to
identify the most valuable customers.
This tool works on vyper environment. To install environment to run `UVyper`, please follow the instructions in
the [Installation Manual](https://github.com/BLEND360/UVyper/blob/idc_dev1/Installation%20Manual.md) file.
This tool has four major steps:

* **Data Preprocessing:** This step is to clean the data and prepare it for the next steps.
* **Clustering:** This step is to cluster the customers based on the features available in the dataset.
* **Postprocessing:** This step is to analyze the clusters and identify the differential features.
* **Playbook and Visualization:** This step is to visualize the clusters and generate the playbook.

#### Files to look for: [UVyper.py](https://github.com/BLEND360/UVyper/blob/idc_dev1/notebooks/UVyper.py)

#### Reference notebook: [UVyper.ipynb](https://github.com/BLEND360/UVyper/blob/idc_dev1/notebooks/UVyper.ipynb)

#### Sample data used : <a href='https://github.com/BLEND360/UVyper/blob/idc_dev1/notebooks/cvs_hcb_member_profiling.csv'>`cvs_hcb_member_profiling.csv`</a>

## Data Preprocessing

* `show_variable_types`: This function recognizes the variable types.

        Method to show the variable types
        :param m: vyper model
        :return: returns the variable types

* `recoding`:  This function removes the excluded variables, encodes binary variables and ordinal variables, removes the
  variables with IQR=0.

        Method to recode the data
        :param bounds: float - this parameter is used to calculate the quantile function for numeric variables. 
                       It determines the range of the bin where the data will be recoded.
        :param min_bin_size: float - this parameter is used to determine whether or not a binary variable should be
                                     split into two separate variables. If the proportion of the most frequent value in 
                                     the binary variable is less than min_bin_size, the variable is left as is. 
                                     Otherwise, two variables are created: one indicating the presence of the most
                                     frequent value, and another indicating the missing values.
        :param dependent_variable: string - this parameter specifies the dependent variable of the dataset.This variable 
                                           is used to train a vyper model to determine which variables should be recoded
        :param ordinal_variables: list - this parameter is used to specify which variables should be treated as ordinal 
                                         variables. If a variable is specified as ordinal, it will be recoded using 
                                         factorization. If it is not specified as ordinal, it will be treated as a 
                                         numeric variable and recoded using the quantile function.
        :return: recoded dataframe, numeric variables, binary variables, categorical variables, ordinal variables

* `category_encoding`: This function encodes the categorical variables.

        Method to encode the categorical variables
        :param category_variables: list - a list of categorical variables in the DataFrame

* `missing_zero_values_table`: This function calculates the missing values and zeros in the dataset.

        Method to generate a table that represents the number and percentage of missing and zero values for each column 
        in the dataset. The table includes columns for the number of zero values, number of missing values, percentage 
        of missing values, total number of zero and missing values, percentage of total zero and missing values, and 
        data type.
        :return: Pandas DataFrame containing the missing and zero values table.

* `drop_missing`: This function drops the variables with more than `threshold` missing values.

        Method to drop the columns with missing values greater than the threshold
        :param threshold: float - (0-1) specifies the threshold value for the proportion of missing values in a column. 
                          Columns with missing values greater than this threshold will be dropped from the dataset.

* `impute_na`: This function imputes the missing values with the mentioned method.

        Method to impute the missing values
        :param columns_list: list -  a list of columns to impute missing values
        :param method: string -  a string specifying the imputation method, which can be one of the following:
                                    "mean": imputes missing values with the mean value of the column
                                    "mode": imputes missing values with the mode value of the column
                                    "bfill": imputes missing values with the next valid value in the col(backward fill)
                                    "ffill": imputes missing values with the prev valid value in the col(forward fill)

* `correlation`: This function calculates the correlation between the variables and drops the highly correlated
  variables.

        Method to remove the highly correlated variables
        :param numerical_variables: list - list of numerical variables to consider for correlation analysis.
        :param ordinal_variables: list - list of ordinal variables to consider for correlation analysis.
        :param threshold: float - threshold value for correlation. Variables with correlation greater than this 
                                  threshold will be removed from the dataset.
        :return: dataframe excluding the highly correlated variables

* `cramers_v_matrix`: This function calculates the Cramer's V matrix and drops the highly correlated variables.

        Method to remove the variables with cramers v value greater than threshold
        :param category_variables: list - list of categorical variables to consider for Cramer's V analysis.
        :param threshold: float - threshold value for Cramer's V. Variables with Cramer's V greater than this threshold 
                                  will be removed from the dataset.
        :return: dataframe - cramers v matrix* `outlier_capping`: This function caps the outliers based on the IQR.

* `outlier_capping`: This function caps the outliers based on the z score.

        Method to cap the outliers
        :param column_list: list - list of columns to consider for outlier capping
        :param thold: float - threshold value for outlier capping (default = 3)
        :return: pd.DataFrame - dataframe with capped outliers

* `outlier_percentages`: This function calculates the percentage of outliers in each variable.

        Method to calculate the percentage of outliers
        :param column_list: list - list of columns to consider for outlier capping
        :param thold: float - threshold value for outlier capping
        :return: pd.DataFrame - dataframe with outlier percentages 

* `standardization`: This function standardizes the variables (standard scaler).

        Method to standardize the data
        :return: pd.DataFrame - standardized dataframe

* `save_preprocessed_data`: This function saves the preprocessed data by taking the path as input.

        Method to save the data to a csv file
        :param filename: str - path to the file

## Clustering

In `__init__` class outlier_per, cramers_matrix, missing_values_table are introduced as parameters. These parameters are
used in generating the playbook.
There are six clustering algorithms available in this tool:

1. **K-Means :**`kmeans_w`

   ```
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
        :param large_data_flag: int - 1 - to save the model and 0 - not to save the model
        :return: ndarray - cluster labels
   ```
   Now there are 2 cases:
    1. If the number of clusters is not given, then this function calculates the optimal number of clusters using
       the
       elbow method and silhouette score by taking minK and maxK as input. If `large_data_flag` is given as 0 then it
       applies `fit_predict` on entire dataset.
       But for larger datasets this might take a lot of time. So, if `large_data_flag` is given as 1 then it
       takes `rand_sample_prop` to get a sample of the dataset. Then it fits the sample data and the model fitted
       with
       sample data is saved as pickle file. Then it applies `predict` on entire dataset.
    2. If the number of clusters is given and if `large_data_flag` is given as 0 then it
       applies `fit_predict` on entire dataset.
       But for larger datasets this might take a lot of time. So, if `large_data_flag` is given as 1 then it
       takes `rand_sample_prop` to get a sample of the dataset. Then it fits the sample data and the model fitted
       with sample data is saved as pickle file. Then it applies `predict` on entire dataset.

   After the clustering is done, we get the cluster labels. Then we calculate the cluster statistics for example
   davies bouldin score, silhouette score, number of customers in each cluster, percentage of customers in each
   cluster, mean of each variable in each cluster.
   <br></br>
2. **K-Medoids :**`kmedoids_w`

   ```
   Method to find the clusters using Kmedoids.
        :param minK: int - The minimum number of clusters to consider.
        :param maxK: int - The maximum number of clusters to consider.
        :param metric: str - optional (default='distortion'). The metric used to quantify the
        quality of clustering. Possible options include distortion, silhouette, calinski_harabasz,
        davies_bouldin, and others.
        :param rand_sample_prop: float - random sampling proportion
        :param filename:  float - path of the pickle file
        :param dataset: float - path of the original dataset
        :param n_clusters: int - no of clusters
        :param large_data_flag: int - 1 - to save the model and 0 - not to save the model
        :return: ndarray - cluster labels
   ```
   Now there are 2 cases:
    1. If the number of clusters is not given, then this function calculates the optimal number of clusters using
       the
       elbow method and silhouette score by taking minK and maxK as input. If `large_data_flag` is given as 0 then it
       applies `fit_predict` on entire dataset.
       But for larger datasets this might take a lot of time. So, if `large_data_flag` is given as 1 then it
       takes `rand_sample_prop` to get a sample of the dataset. Then it fits the sample data and the model fitted
       with
       sample data is saved as pickle file. Then it applies `predict` on entire dataset.
    2. If the number of clusters is given and if `large_data_flag` is given as 0 then it
       applies `fit_predict` on entire dataset.
       But for larger datasets this might take a lot of time. So, if `large_data_flag` is given as 1 then it
       takes `rand_sample_prop` to get a sample of the dataset. Then it fits the sample data and the model fitted
       with sample data is saved as pickle file. Then it applies `predict` on entire dataset.

   After the clustering is done, we get the cluster labels. Then we calculate the cluster statistics for example
   davies bouldin score, silhouette score, number of customers in each cluster, percentage of customers in each
   cluster, mean of each variable in each cluster.
   <br></br>
3. **Mini Batch K-Means :**`minibatchkmeans_w`

   ```
   Method to find the clusters using minibatchkmeans.
        :param minK: int - The minimum number of clusters to consider.
        :param maxK: int - The maximum number of clusters to consider.
        :param metric: str - optional (default='distortion'). The metric used to quantify the
        quality of clustering. Possible options include distortion, silhouette, calinski_harabasz,
        davies_bouldin, and others.
        :param rand_sample_prop: float - random sampling proportion
        :param filename:  float - path of the pickle file
        :param dataset: float - path of the original dataset
        :param n_clusters: int - no of clusters
        :param large_data_flag: int - 1 - to save the model and 0 - not to save the model
        :return: ndarray - cluster labels
   ```
   Now there are 2 cases:
    1. If the number of clusters is not given, then this function calculates the optimal number of clusters using
       the
       elbow method and silhouette score by taking minK and maxK as input. If `large_data_flag` is given as 0 then it
       applies `fit_predict` on entire dataset.
       But for larger datasets this might take a lot of time. So, if `large_data_flag` is given as 1 then it
       takes `rand_sample_prop` to get a sample of the dataset. Then it fits the sample data and the model fitted
       with
       sample data is saved as pickle file. Then it applies `predict` on entire dataset.
    2. If the number of clusters is given and if `large_data_flag` is given as 0 then it
       applies `fit_predict` on entire dataset.
       But for larger datasets this might take a lot of time. So, if `large_data_flag` is given as 1 then it
       takes `rand_sample_prop` to get a sample of the dataset. Then it fits the sample data and the model fitted
       with sample data is saved as pickle file. Then it applies `predict` on entire dataset.

   After the clustering is done, we get the cluster labels. Then we calculate the cluster statistics for example
   davies bouldin score, silhouette score, number of customers in each cluster, percentage of customers in each
   cluster, mean of each variable in each cluster.
   <br></br>
4. **Hierarchical :**`hierarchical_w`

   ```
   Method to perform hierarchical clustering.
        :param param_grid: dict - parameters grid
        :param folds: int - number of folds
        :param n_iter: int - number of iterations
        :param rand_sample_prop: float - random sample proportion
        :param dataset: str - path to the original dataset
        :param n_neighbors: int - number of neighbors
        :param n_clusters: int - number of clusters
        :param linkage: str - linkage
        :param affinity: str - affinity
        :return: ndarray - cluster labels
   ```
   Now there are 2 cases:
    1. If n_clusters or linkage or affinity is none it performs the randomized grid search to find the optimal
       parameters.

    2. Else we consider parameters that are given.
   
    After the hyperparameter tuning is done, we get the optimal parameters. Then we provide size of the sample
       dataset to `rand_sample_prop` and it takes a sample of the dataset. Then we fit and predict on this sample
       dataset, and we pass this sample dataset along with the labels we get, into KNeighborsClassifier where we
       consider `n_neighbors` as default 5. Then we predict on entire dataset .
       
       After the clustering is done, we get the cluster labels. Then we calculate the cluster statistics for example
       davies bouldin score, silhouette score, number of customers in each cluster, percentage of customers in each
       cluster, mean of each variable in each cluster.
       <br></br>
       **Note:** In the parameters for linkage=='ward' only affinity=='euclidean' is allowed.
       <br></br>
5. **GMM:** `gmm_w`

   ```
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
        :param large_data_flag: int - 1 - to save the model and 0 - not to save the model
        :return: ndarray - The cluster labels.
   ```
   Now there are 2 cases:
    1. If the number of components or covariance type or init_params, is not given, it performs
       randomized grid search to get optimal parameters.
       If `large_data_flag` is given as 0 then it
       applies `fit_predict` on entire dataset.
       But for larger datasets this might take a lot of time. So, if `large_data_flag` is given as 1 then it
       takes `rand_sample_prop` to get a sample of the dataset. Then it fits the sample data and the model fitted
       with
       sample data is saved as pickle file. Then it applies `predict` on entire dataset.
    2. If all three parameters are given and if `large_data_flag` is given as 0 then it
       applies `fit_predict` on entire dataset.
       But for larger datasets this might take a lot of time. So, if `large_data_flag` is given as 1 then it
       takes `rand_sample_prop` to get a sample of the dataset. Then it fits the sample data and the model fitted
       with sample data is saved as pickle file. Then it applies `predict` on entire dataset.
       <br></br>
6. **Birch(Balanced Iterative Reducing and Clustering using Hierarchies):** `birch_w`

   ```
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
        :param large_data_flag: int - 1 - to save the model and 0 - not to save the model
        :return: ndarray - The cluster labels.
   ```
   Now there are 2 cases:
    1. If the number of clusters or threshold or branching factor is not given, it performs randomized grid search to
       get optimal
       parameters.
       If `large_data_flag` is given as 0 then it
       applies `fit_predict` on entire dataset.
       But for larger datasets this might take a lot of time. So, if `large_data_flag` is given as 1 then it
       takes `rand_sample_prop` to get a sample of the dataset. Then it fits the sample data and the model fitted
       with
       sample data is saved as pickle file. Then it applies `predict` on entire dataset.
    2. If all three parameters are given and if `large_data_flag` is given as 0 then it
       applies `fit_predict` on entire dataset.
       But for larger datasets this might take a lot of time. So, if `large_data_flag` is given as 1 then it
       takes `rand_sample_prop` to get a sample of the dataset. Then it fits the sample data and the model fitted
       with sample data is saved as pickle file. Then it applies `predict` on entire dataset.
       <br></br>

* `get_models_summary` recommends you the best model and gives you the table with ranks of the models based on the two
  evaluation metrics considered:
  Davies Bouldin Score and Silhouette Score. Davies bouldin score is a measure of the average similarity between
  clusters. The lower the score, the better the clustering. Silhouette score is a measure of how similar an object is
  to its own cluster compared to other clusters. The higher the score, the better the clustering.

## Post Processing

* `post_process` method is used to find the `n` most differential features. This was done using random forest
  classifier. The features were ranked based on their importance. The features with the highest importance were
  considered as the most
  differential features. This features doesn't include categorical features.
  <br></br>
  ```
  Method to perform post-processing of the clustering results
        :param recommended_model: str - name of the recommended model
        :param org_dataset: str - path to the original dataset
        :param preprocessed_dataset: str - path to the preprocessed dataset
        :param dependent_variable: str - name of the dependent variable
        :param filename: str - filename to save the parallel coordinates plot (with .png extension)
        :param kmeans_cluster_labels: np.ndarray - cluster labels of KMeans (optional)
        :param hierarchical_cluster_labels: np.ndarray - cluster labels of Hierarchical (optional)
        :param gmm_cluster_labels: np.ndarray - cluster labels of GMM (optional)
        :param birch_cluster_labels: np.ndarray - cluster labels of Birch (optional)
        :param kmedoids_cluster_labels: np.ndarray - cluster labels of Kmedoids (optional)
        :param mini_batch_kmeans_cluster_labels: np.ndarray - cluster labels of Mini Batch Kmeans (optional)
        :param n_variables: int - number of differential factors to be considered
        :return: pd.DataFrame, pd.DataFrame - reduced preprocessed dataset to 2 dimensions using PCA, dataframe with differential factors and cluster labels
  ```

## Playbook and Visualization

* Playbook contains 7 sheets:
    1. **Summary:** This sheet contains the vyper generated summary of the dataset
    2. **Distribution:** This sheet contains the distribution of data in each cluster of each model and the ranked score
       table of models
    3. **Describe:** This sheet contains the statistics of original dataset.
    4. **Analysis:** This sheet contains the missing value percentage, outlier percentage, cramers_matrix
    5. **Charts:** This sheet contains the scatter plot of the reduced dataset using PCA and the parallel coordinates
       plot of the differential features.
    6. **Cluster wise Feature statistics:** This sheet contains the statistics of each cluster of best model.
    7. **Cluster wise Feature analysis:** This sheet contains the analysis of each cluster of best model.
       <br></br>
  ```
  Method to generate playbook
        :param filename: str - filename to save the playbook
        :param org_dataset: str - path to the original dataset
        :param dependent_variable: str - dependent variable
        :param pca: pd.DataFrame - pca dataframe of preprocess dataset
        :param im: str - path to the parallel plot png file (with .png extension
        :param diff_factors_dataframe: differential factors dataframe with best model cluster labels
        :param to_delete: int - to delete the parallel plot png file after generating playbook
        :param kmeans_cluster_labels: np.ndarray - kmeans cluster labels
        :param hierarchical_cluster_labels: np.ndarray - hierarchical cluster labels
        :param gmm_cluster_labels: np.ndarray - gmm cluster labels
        :param birch_cluster_labels: np.ndarray - birch cluster labels
        :param kmedoids_cluster_labels: np.ndarray - kmedoids cluster labels
        :param mini_batch_kmeans_cluster_labels: np.ndarray - mini batch kmeans cluster labels
        :return:
  ```