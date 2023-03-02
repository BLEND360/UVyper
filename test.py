from my_library.utils import preprocessing as pp

# pps = pp.Preprocessing('maurices.csv')
pps = pp.Preprocessing('segmentation_nonkroger_dataset_v3.csv')
# print(pps.show_variable_types(pps.vyper('indiv_id_mcvp')))
# a, B, nv, bv = pps.recoding(0.98, 0.1, 'indiv_id_mcvp', )
a, B, nv, bv = pps.recoding(0.98, 0.1, 'person_id', )
num_vars = [s + '_processed' for s in nv]
bin_vars = [s + '_ind' for s in bv]
pps.impute_na(num_vars, 'mean')
pps.impute_na(bin_vars, 'mode')
pps.correlation(num_vars, 0.8)
pps.outlier_detection()  # comment for goodrx data
pps.standardization()
pps.print_shape()
# pps.tocsv('maurices')
pps.tocsv('segmentation_nonkroger_dataset_v31')

# pip install ortools==9.3.10459
from my_library.algorithms.k_means import k_means as km

# kms = km.Kmeans('maurices_preprocessed.csv')
kms = km.Kmeans('segmentation_nonkroger_dataset_v3_preprocessed.csv')
kms.Kmeans_elbow_plot(2, 10)
kms.kmeans_minClusterSize(5, 500)
kms.kmeans_maxClusterSize(5, 500)
clus = kms.kmeans(5, 1000, 4000)
pca = kms.pca(clus)
kms.scatter_plot(pca)
