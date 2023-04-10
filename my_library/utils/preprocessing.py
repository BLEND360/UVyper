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
from scipy.stats import chi2, chi2_contingency
from sklearn.preprocessing import StandardScaler
from varclushi import VarClusHi
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)


class Preprocessing:
    def __init__(self, data: str):
        """
        __init__ method for the Preprocessing class
        :param data: string - path to the data
        """
        self.df = pd.read_csv(data)

    def get_df(self):
        """
        returns the data
        :return: dataframe
        """
        return self.df

    def get_shape(self):
        """
        prints the shape of the data
        :return: tuple
        """
        return self.df.shape

    def vyper(self, dependent_variable: str):
        """
        Method to insert data into vyper
        :param dependent_variable: string
        :return: vyper model
        """
        m = Model(
            data=self.df,
            dependent_variable=dependent_variable,
            na_drop_threshold=0.5,  # set here when variables should be dropped 1 = none dropped
            training_percentage=0.7,  # set here % of dataset that should be training
            model_type="linear",
        )
        return m

    @staticmethod
    def show_variable_types(m):
        """
        Method to show the variable types
        :param m: vyper model
        :return: returns the variable types
        """
        return m.variables.show_types()

    @staticmethod
    def quantile_function(a: list, bounds: float):
        """
        Method to calculate the quantile function
        :param a: list - a list of variables for which the quantile function is to be calculated
        :param bounds: float - a float representing the bounds around the median for which the quantile function is to be calculated
        :return: list - returns the lower and upper bound
        """
        lower_bound = np.quantile(a, q=(0.5 - bounds / 2), interpolation="nearest")
        upper_bound = np.quantile(a, q=(0.5 + bounds / 2), interpolation="nearest")
        final_bounds = [lower_bound, upper_bound]

        return final_bounds

    def recoding(self, bounds: float, min_bin_size: float, dependent_variable: str,
                 ordinal_variables: list = None):
        """
        Method to recode the data
        :param bounds: float - this parameter is used to calculate the quantile function for numeric variables. It determines the range of the bin where the data will be recoded.
        :param min_bin_size: float - this parameter is used to determine whether or not a binary variable should be split into two separate variables. If the proportion of the most frequent value in the binary variable is less than min_bin_size, the variable is left as is. Otherwise, two variables are created: one indicating the presence of the most frequent value, and another indicating the missing values.
        :param dependent_variable: string - this parameter specifies the dependent variable of the dataset. This variable is used to train a vyper model to determine which variables should be recoded.
        :param ordinal_variables: list - this parameter is used to specify which variables should be treated as ordinal variables. If a variable is specified as ordinal, it will be recoded using factorization. If it is not specified as ordinal, it will be treated as a numeric variable and recoded using the quantile function.
        :return: dataframe, list, list, set, list
        """
        m = self.vyper(dependent_variable)
        original_variables = m.data.columns.to_list()
        excluded_variables = m.variables.get_excluded_variables()
        category_variables = m.variables.get_categorical_variables()
        numeric_variables = m.variables.get_numeric_variables()
        binary_variables = m.variables.get_binary_variables()
        keep_variables = original_variables
        if ordinal_variables is None:
            ordinal_variables = []
        else:
            ordinal_variables = ordinal_variables
            for var in ordinal_variables:
                if var in excluded_variables:
                    excluded_variables.remove(var)
                elif var in category_variables:
                    category_variables.remove(var)
                elif var in numeric_variables:
                    numeric_variables.remove(var)
                elif var in binary_variables:
                    binary_variables.remove(var)

        nv = list(numeric_variables)
        bv = list(binary_variables)
        ov = list(ordinal_variables)

        for var in excluded_variables:
            if var in self.df.columns:
                keep_variables.remove(var)
                self.df.drop(var, axis=1, inplace=True)
        print("Excluded variables (vyper): ", excluded_variables)
        for var in numeric_variables:
            if var in self.df.columns:
                Q3 = np.quantile(self.df[[var]], 0.75)
                Q1 = np.quantile(self.df[[var]], 0.25)
                IQR = Q3 - Q1

                if IQR == 0:
                    keep_variables.remove(var)
                    nv.remove(var)
                    self.df.drop(var, axis=1, inplace=True)
                    print(
                        "The following variables is excluded due to inter quantile equal 0:"
                        + var
                    )

                else:
                    bnds = self.quantile_function(
                        [var for var in list(self.df[var]) if not math.isnan(var)],
                        bounds=bounds,
                    )
                    self.df[var + "_processed"] = pd.Series(
                        np.minimum(np.maximum(self.df[var], bnds[0]), bnds[1])
                    )
                    keep_variables.remove(var)
                    nv.remove(var)
                    keep_variables.append(var + "_processed")
                    nv.append(var + "_processed")
                    self.df = self.df[keep_variables]

        for var in ordinal_variables:
            if var in self.df.columns:
                training_data = self.df[[var]].drop_duplicates()
                training_data[var + "_processed"] = (
                        pd.factorize(training_data[var], sort=True)[0] + 1
                )
                self.df = self.df.merge(training_data, on=var)
                keep_variables.remove(var)
                ov.remove(var)
                keep_variables.append(var + "_processed")
                ov.append(var + "_processed")
                self.df = self.df[keep_variables]

        N = self.df.shape[0]

        for var in binary_variables:
            if var in self.df.columns:
                tab = self.df[var].value_counts(ascending=False)
                counter2 = 0
                counter1 = tab.iloc[1]
                counter2 = counter1 + counter2
                N_missing = sum(self.df[var].isna())

                if N_missing / N >= min_bin_size:
                    self.df[var + "_missing_ind"] = np.where(
                        self.df[var].isna() == 1, 1, 0
                    )

                if counter2 / (N - N_missing) >= min_bin_size:
                    self.df[var + "_ind"] = np.where(self.df[var] == tab.index[1], 1, 0)
                self.df = self.df.drop([var], axis=1)
                bv.remove(var)
                bv.append(var + "_ind")

        return self.df, nv, bv, category_variables, ov

    def category_encoding(self, category_variables: list):
        """
        Method to encode the categorical variables
        :param category_variables: list - a list of categorical variables in the DataFrame
        """
        temp = self.df[category_variables]
        temp = pd.get_dummies(
            temp, prefix=category_variables, columns=category_variables
        )
        self.df.drop(category_variables, axis=1, inplace=True)
        self.df = pd.concat([self.df, temp], axis=1)

    def missing_zero_values_table(self):
        """
        Method to generate a table that represents the number and percentage of missing and zero values for each column in the dataset. The table includes columns for the number of zero values, number of missing values, percentage of missing values, total number of zero and missing values, percentage of total zero and missing values, and data type.
        :return: Pandas DataFrame containing the missing and zero values table.
        """

        zero_val = (self.df == 0.00).astype(int).sum(axis=0)
        mis_val = self.df.isnull().sum()
        mis_val_percent = (self.df.isnull().sum() / len(self.df)) * 100
        missing_table = pd.concat([zero_val, mis_val, mis_val_percent], axis=1)
        missing_table = missing_table.rename(
            columns={0: "Zero Values", 1: "Missing Values", 2: "% of Total Values"}
        )
        missing_table["Total Zero + Missing Values"] = (
                missing_table["Zero Values"] + missing_table["Missing Values"]
        )
        missing_table["% Total Zero + Missing Values"] = (missing_table["Total Zero + Missing Values"] / len(
            self.df)) * 100
        missing_table["Data Type"] = self.df.dtypes
        missing_table = (
            missing_table[missing_table.iloc[:, 1] != 0]
            .sort_values("% of Total Values", ascending=False)
            .round(1)
        )

        return missing_table

    def drop_missing(self, threshold: float):
        """
        Method to drop the columns with missing values greater than the threshold
        :param threshold: float - specifies the threshold value for the proportion of missing values in a column. Columns with missing values greater than this threshold will be dropped from the dataset.
        """
        drop_list = []
        for cols in self.df.columns:
            if (self.df.loc[:, cols].isnull().sum() / len(self.df)) > int(threshold):
                drop_list.append(cols)
        self.df.drop(drop_list, axis=1, inplace=True)

    def impute_na(self, columns_list: list, method: str):  # mean, mode,  bfill, ffill
        """
        Method to impute the missing values
        :param columns_list: list -  a list of columns to impute missing values
        :param method: string -  a string specifying the imputation method, which can be one of the following:
                                    "mean": imputes missing values with the mean value of the column
                                    "mode": imputes missing values with the mode value of the column
                                    "bfill": imputes missing values with the next valid value in the column (backward fill)
                                    "ffill": imputes missing values with the previous valid value in the column (forward fill)
        """
        if method == "mean":
            for i in columns_list:
                if i in self.df.columns:
                    self.df[i].fillna(self.df[i].mean(), inplace=True)
        if method == "mode":
            for i in columns_list:
                if i in self.df.columns:
                    self.df[i].fillna(self.df[i].mode()[0], inplace=True)
        if method == "bfill":
            for i in columns_list:
                if i in self.df.columns:
                    self.df[i].fillna("bfill", inplace=True)
        if method == "ffill":
            for i in columns_list:
                if i in self.df.columns:
                    self.df[i].fillna("ffill", inplace=True)

    def correlation(self, numerical_variables: list, ordinal_variables: list = None, threshold: float = 0.8):
        """
        Method to remove the highly correlated variables
        :param numerical_variables: list - list of numerical variables to consider for correlation analysis.
        :param ordinal_variables: list - list of ordinal variables to consider for correlation analysis.
        :param threshold: float - threshold value for correlation. Variables with correlation greater than this threshold will be removed from the dataset.
        :return: dataframe
        """
        if ordinal_variables is None:
            ordinal_variables = []
        df_copy = self.df.copy()
        cols = self.df.columns
        newcol = []
        for col in numerical_variables + ordinal_variables:
            if col in cols:
                newcol.append(col)
        df_copy = df_copy[newcol]
        list1 = list()
        col_corr = set()  # Set of all the names of deleted columns
        corr_matrix = df_copy.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if (corr_matrix.iloc[i, j] >= threshold) and (
                        corr_matrix.columns[j] not in col_corr
                ):
                    colname = corr_matrix.columns[i]  # getting the name of column
                    col_corr.add(colname)
                    if colname in self.df.columns:
                        list1.append(colname)
                        del self.df[colname]  # deleting the column from the self.df
        print("List of columns removed due to high correlation: ", list1)
        return pd.DataFrame(self.df)

    @staticmethod
    def cramers_V(variable_1: str, variable_2: str):
        """
          Method to calculate the Cramer's V statistic for categorical-categorical association.
          :param variable_1: string - categorical variable
          :param variable_2: string - categorical variable
          :return: float
          """
        crosstab = np.array(pd.crosstab(variable_1, variable_2, rownames=None, colnames=None))  # Cross table building
        stat = chi2_contingency(crosstab)[0]  # Keeping of the test statistic of the Chi2 test
        obs = np.sum(crosstab)  # Number of observations
        mini = min(crosstab.shape) - 1  # Take the minimum value between the columns and the rows of the cross table
        return stat / (obs * mini)

    def cramers_v_matrix(self, category_variables: list, threshold: float = 0.1):
        """
        Method to remove the variables with cramers v value greater than threshold
        :param category_variables: list - list of categorical variables to consider for Cramer's V analysis.
        :param threshold: float - threshold value for Cramer's V. Variables with Cramer's V greater than this threshold will be removed from the dataset.
        :return: list
        """
        data_df_orig = self.df[category_variables]
        rows = []

        for var1 in data_df_orig:
            col = []
            for var2 in data_df_orig:
                cramers = self.cramers_V(data_df_orig[var1], data_df_orig[var2])  # Cramer's V test
                col.append(round(cramers, 2))  # Keeping of the rounded value of the Cramer's V
            rows.append(col)

        cramers_results = np.array(rows)
        df_matrix = pd.DataFrame(cramers_results, columns=data_df_orig.columns, index=data_df_orig.columns)

        list1 = list()
        col_cramers = set()  # Set of all the names of deleted columns

        for i in range(len(df_matrix.columns)):
            for j in range(i):
                if (df_matrix.iloc[i, j] >= threshold) and (df_matrix.columns[j] not in col_cramers):
                    colname = df_matrix.columns[i]  # getting the name of column
                    col_cramers.add(colname)
                    if colname in data_df_orig.columns:
                        list1.append(colname)
                        del data_df_orig[colname]  # deleting the column from the dataset
                        del self.df[colname]
                        category_variables.remove(colname)
        # self.df=data_df_orig
        print("List of columns removed due to higher than the threshold cramers v value: ", list1)

    def calculateMahalanobis(self, cov=None, alpha=0.01):
        """
        Method to calculate the mahalanobis distance and p values
        :param cov:
        :param alpha:
        :return:
        """

        y_mu = self.df - np.mean(self.df)
        if not cov:
            cov = np.cov(self.df.values.T)
        inv_covmat = np.linalg.inv(cov)
        left = np.dot(y_mu, inv_covmat)
        mahal = np.dot(left, y_mu.T)
        self.df['Mahalanobis'] = mahal.diagonal()
        # calculate p values Degrees of Freedom is number of columns
        self.df['p'] = 1 - chi2.cdf(self.df['Mahalanobis'], self.df.shape[1])
        # remove rows with p < alpha
        to_remove = len(self.df[self.df.p < alpha])
        print("Number of rows removed from dataset: ", to_remove)
        self.df = self.df[self.df.p > alpha]
        self.df.drop('Mahalanobis', axis=1, inplace=True)
        self.df.drop('p', axis=1, inplace=True)
        return self.df

    def outlier_capping(self, column_list: list, thold: float = 3):
        """
        Method to cap the outliers
        :param column_list: list - list of columns to consider for outlier capping
        :param thold: float - threshold value for outlier capping
        :return: dataframe
        """
        for col in column_list:
            mu = self.df[col].mean()
            sigma = self.df[col].std()
            scaled_data = (self.df[col] - mu) / sigma
            upper_limit = scaled_data.mean() + (thold * scaled_data.std())
            lower_limit = scaled_data.mean() - (thold * scaled_data.std())
            capped_data = np.where(scaled_data > upper_limit, upper_limit,
                                   np.where(scaled_data < lower_limit, lower_limit, scaled_data))
            self.df[col] = capped_data * sigma + mu
        return self.df

    def standardization(self):
        """
        Method to standardize the data
        :return: dataframe
        """
        scaler = StandardScaler()
        scaled = scaler.fit_transform(self.df)
        self.df = pd.DataFrame(scaled, columns=self.df.columns)
        return self.df

    def var_clustering(self, maxeigval2: float = 1, maxclus=None):
        """
        Method to cluster the variables
        :param maxeigval2: float - threshold value for eigen value
        :param maxclus:
        :return: dataframe, list, int
        """
        var_clust_model = VarClusHi(self.df, maxeigval2=maxeigval2, maxclus=maxclus)
        var_clust_model.varclus()
        var_to_keep = list(
            var_clust_model.rsquare.sort_values(by=['Cluster', 'RS_Ratio']).groupby('Cluster').first().Variable)
        return var_clust_model.rsquare, var_to_keep, len(var_to_keep)
        # from here need to get the variable within a cluster with the lowest RS_Ratio and pick that one
