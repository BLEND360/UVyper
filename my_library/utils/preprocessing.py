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


class Preprocessing:
    def __init__(self, data):  # to read the data
        self.df = pd.read_csv(data)

    def print(self):  # prints the data
        print(self.df)
        # return self.df

    def get_df(self):  # returns the data
        return self.df

    def print_shape(self):  # prints the shape of the data
        print(self.df.shape)
        return self.df.shape

    def vyper(self, dv):  # model vyper
        m = Model(
            data=self.df,
            dependent_variable=dv,
            na_drop_threshold=0.5,  # set here when variables should be dropped 1 = none dropped
            training_percentage=0.7,  # set here % of dataset that should be training
            model_type="linear",
        )
        return m

    @staticmethod
    def show_variable_types(m):  # shows the variable types
        return m.variables.show_types()

    @staticmethod
    def quantile_function(a, bounds):
        lower_bound = np.quantile(a, q=(0.5 - bounds / 2), interpolation="nearest")
        upper_bound = np.quantile(a, q=(0.5 + bounds / 2), interpolation="nearest")
        final_bounds = [lower_bound, upper_bound]

        return final_bounds

    def recoding(self, bounds, min_bin_size, dv,
                 ordinal_variables=None):  # removed excluded_variables,numeric_variables,binary_variables,category_variables as parameters and introduced dependent variable as parameter

        if ordinal_variables is None:
            ordinal_variables = []
        m = self.vyper(dv)
        original_variables = m.data.columns.to_list()
        excluded_variables = m.variables.get_excluded_variables()
        category_variables = m.variables.get_categorical_variables()
        numeric_variables = m.variables.get_numeric_variables()
        binary_variables = m.variables.get_binary_variables()
        keep_variables = original_variables

        for var in excluded_variables:
            if var in self.df.columns:
                keep_variables.remove(var)
                self.df.drop(var, axis=1, inplace=True)

        for var in numeric_variables:
            if var in self.df.columns:
                Q3 = np.quantile(self.df[[var]], 0.75)
                Q1 = np.quantile(self.df[[var]], 0.25)
                IQR = Q3 - Q1

                if IQR == 0:
                    keep_variables.remove(var)
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
                    keep_variables.append(var + "_processed")
                    self.df = self.df[keep_variables]

        for var in ordinal_variables:
            if var in self.df.columns:
                training_data = self.df[[var]].drop_duplicates()
                training_data[var + "_processed"] = (
                        pd.factorize(training_data[var], sort=True)[0] + 1
                )
                self.df = self.df.merge(training_data, on=var)
                keep_variables.remove(var)
                keep_variables.append(var + "_processed")
                self.df = self.df[keep_variables]

        data_df_orig = self.df
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
                    data_df_orig[var].fillna("TnImissing", inplace=True)

                else:
                    data_df_orig[var].fillna(tab.index[0], inplace=True)

                if counter2 / (N - N_missing) >= min_bin_size:
                    self.df[var + "_ind"] = np.where(self.df[var] == tab.index[1], 1, 0)
                self.df = self.df.drop([var], axis=1)

        self.df = pd.get_dummies(
            self.df, prefix=category_variables, columns=category_variables
        )

        for var in category_variables:
            if var in self.df.columns:  # changed from data_df_orig.columns to self.df.columns
                N_missing = sum(data_df_orig[var].isna())
                tab = data_df_orig[var].value_counts(ascending=False)

                if N_missing / N >= min_bin_size:
                    self.df[var + "_missing_ind"] = self.df[var].isna().astype(int)
                    data_df_orig[var].fillna("TnImissing", inplace=True)
                else:
                    data_df_orig[var].fillna(tab.index[0], inplace=True)

                self.df = self.df.drop([var + "_" + tab.index[0]], axis=1)
                counter2 = 0

                for ii in range(1, (len(tab) - 1)):
                    counter1 = tab.iloc[ii]
                    counter2 = counter1 + counter2
                    if counter2 / (N - N_missing) < min_bin_size:
                        self.df = self.df.drop([var + "_" + tab.index[ii]], axis=1)
                        data_df_orig[var] = np.where(
                            data_df_orig[var] == tab.index[ii],
                            tab.index[0],
                            data_df_orig[var],
                        )
                        counter2 = 0

        return self.df, data_df_orig, numeric_variables, binary_variables  # added numeric_variables, binary_variables as return values

    def missing_zero_values_table(self):

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

    def drop_missing(self, thold):
        drop_list = []
        for cols in self.df.columns:
            if (self.df.loc[:, cols].isnull().sum() / len(self.df)) > int(thold):
                drop_list.append(cols)
        self.df.drop(drop_list, axis=1, inplace=True)

    def impute_na(self, cols, mth):  # mean, mode,  bfill, ffill
        if mth == "mean":
            for i in cols:
                if i in self.df.columns:
                    self.df[i].fillna(self.df[i].mean(), inplace=True)
        if mth == "mode":
            for i in cols:
                if i in self.df.columns:
                    self.df[i].fillna(self.df[i].mode()[0], inplace=True)
        if mth == "bfill":
            for i in cols:
                if i in self.df.columns:
                    self.df[i].fillna("bfill", inplace=True)
        if mth == "ffill":
            for i in cols:
                if i in self.df.columns:
                    self.df[i].fillna("ffill", inplace=True)

    def correlation(self, nv, threshold):  # nv = numeric variables which is returned from recoding function
        cols = self.df.columns
        newcol = []
        for col in nv:
            if col in cols:
                newcol.append(col)
        self.df = self.df[newcol]
        list1 = list()
        col_corr = set()  # Set of all the names of deleted columns
        corr_matrix = self.df.corr()
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
        print("List of columns removed from the self.df : ", list1)
        return pd.DataFrame(self.df)

    # def outlier_detection(self):  # IQR method has been used instead of mahalanobis distance
    #     q1 = self.df.quantile(0.25)
    #     q3 = self.df.quantile(0.75)
    #     iqr = q3 - q1
    #     self.df = self.df[
    #         ~((self.df < (q1 - 1.5 * iqr)) | (self.df > (q3 + 1.5 * iqr)))
    #     ]
    #     self.df = self.df.dropna().reset_index(drop=True)
    #     return self.df
#
    def outlier_capping(self, col, thold):

        mu = self.df[col].mean()
        sigma = self.df[col].std()
        scaled_data = (self.df[col] - mu) / sigma
        upper_limit = scaled_data.mean() + (thold * scaled_data.std())
        lower_limit = scaled_data.mean() - (thold * scaled_data.std())
        capped_data = np.where(scaled_data > upper_limit, upper_limit,
                               np.where(scaled_data < lower_limit, lower_limit, scaled_data))
        self.df[col] = capped_data * sigma + mu
        return self.df

    def oc(self, thold=3):
        for col in self.df.columns:
            self.outlier_capping(col, thold)
        return self.df

    def standardization(self):
        scaler = StandardScaler()
        scaled = scaler.fit_transform(self.df)
        self.df = pd.DataFrame(scaled, columns=self.df.columns)
        return self.df

    # def var_clustering(self, maxeigval2=1, maxclus=None):
    #     var_clust_model = VarClusHi(self.df, maxeigval2=maxeigval2, maxclus=maxclus)
    #     var_clust_model.varclus()
    #     var_to_keep = list(
    #         var_clust_model.rsquare.sort_values(by=['Cluster', 'RS_Ratio']).groupby('Cluster').first().Variable)
    #     return (var_clust_model.rsquare,
    #             var_to_keep)
    #             #from here need to get the variable within a cluster with the lowest RS_Ratio and pick that one

    def tocsv(self,
              filename):
        # save the preprocessed data to csv file with the name of the original file excluding .csv  + _preprocessed
        return self.df.to_csv(filename + '_preprocessed.csv', index=False)
