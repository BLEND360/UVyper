from my_library.utils import preprocessing as pp

pps = pp.Preprocessing('maurices.csv')
a, B, nv, bv = pps.recoding(0.98, 0.1, 'indiv_id_mcvp', )
num_vars = [s + '_processed' for s in nv]
bin_vars = [s + '_ind' for s in bv]
pps.impute_na(num_vars, 'mean')
pps.impute_na(bin_vars, 'mode')
pps.correlation(num_vars, 0.8)
pps.oc(6)
pps.standardization()
pps.print_shape()
pps.tocsv('maurices_oc')
# pip install ortools==9.3.10459
from my_library.algorithms.k_means import k_means as km

kms = km.Kmeans('maurices_oc_preprocessed.csv')
kms.Kmeans_elbow_plot(2, 10)
kms.kmeans_minClusterSize(5, 10000)
kms.kmeans_maxClusterSize(5, 10000)
clus = kms.kmeans(5, 5000, 30000)
pca = kms.pca(clus)
kms.scatter_plot(pca)
