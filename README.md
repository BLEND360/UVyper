### Download this <a href="https://github.com/BLEND360/UVyper/blob/idc_dev1/UVyper.yaml">file</a> and use the below mentioned command to install the `uvyp` environment

```
conda env create --file UVyper.yml
```
#### Sample data used : <a href='https://github.com/BLEND360/UVyper/blob/idc_dev1/notebooks/cvs_hcb_member_profiling.csv'>`cvs_hcb_member_profiling.csv`</a>

[//]: # (#### <a href = 'https://github.com/BLEND360/UVyper/tree/idc_dev1/my_library'>`my_library`</a> folder contains:)

[//]: # ()
[//]: # (- <a href = 'https://github.com/BLEND360/UVyper/tree/idc_dev1/my_library/utils'>`utils`</a> folder contains <a href = 'https://github.com/BLEND360/UVyper/tree/idc_dev1/my_library/utils/preprocessing.py'>`preprocessing.py`</a> file which contains the implementation of preprocessing)

[//]: # (  functions)

[//]: # (- <a href = 'https://github.com/BLEND360/UVyper/tree/idc_dev1/my_library/algorithms'>`algorithms`</a> folder which)

[//]: # (  contains all the)

[//]: # (  clustering algorithms:)

[//]: # (    - <a href = 'https://github.com/BLEND360/UVyper/tree/idc_dev1/my_library/algorithms/birch'>`birch`</a> folder)

[//]: # (      contains <a href = 'https://github.com/BLEND360/UVyper/tree/idc_dev1/my_library/algorithms/birch/birch.py'>`birch.py`</a>)

[//]: # (      file which contains)

[//]: # (      the implementation of birch algorithm)

[//]: # (    - <a href = 'https://github.com/BLEND360/UVyper/tree/idc_dev1/my_library/algorithms/dbscan'>`dbscan`</a> folder)

[//]: # (      contains)

[//]: # (      <a href = 'https://github.com/BLEND360/UVyper/tree/idc_dev1/my_library/algorithms/dbscan/dbscan.py'> `dbscan.py`</a>)

[//]: # (      file which contains the implementation of dbscan algorithm)

[//]: # (    - <a href = 'https://github.com/BLEND360/UVyper/tree/idc_dev1/my_library/algorithms/gmm'>`gmm`</a> folder)

[//]: # (      contains <a href = 'https://github.com/BLEND360/UVyper/tree/idc_dev1/my_library/algorithms/gmm/gmm.py'>`gmm.py`</a>)

[//]: # (      file which contains the)

[//]: # (      implementation of gmm algorithm)

[//]: # (    - <a href = 'https://github.com/BLEND360/UVyper/tree/idc_dev1/my_library/algorithms/hierarchical'>`hierarchical`</a>)

[//]: # (      folder)

[//]: # (      contains <a href = 'https://github.com/BLEND360/UVyper/tree/idc_dev1/my_library/algorithms/hierarchical/hierarchical.py'>`hierarchical.py`</a>)

[//]: # (      file which)

[//]: # (      contains the implementation of kmeans algorithm)

[//]: # (    - <a href = 'https://github.com/BLEND360/UVyper/tree/idc_dev1/my_library/algorithms/k_means'>`k_means`</a> folder)

[//]: # (      contains <a href = 'https://github.com/BLEND360/UVyper/tree/idc_dev1/my_library/algorithms/k_means/k_means.py'>`k_means.py`</a>)

[//]: # (      file which contains)

[//]: # (      the implementation of kmeans algorithm)

[//]: # (- <a href = 'https://github.com/BLEND360/UVyper/tree/idc_dev1/notebooks'>`notebooks`</a> folder contains all)

[//]: # (  the jupyter)

[//]: # (  notebooks:)

[//]: # (    - <a href = 'https://github.com/BLEND360/UVyper/blob/idc_dev1/notebooks/preprocessing.ipynb'>`preprocessing.ipynb`</a> works on sample data that was taken and performs preprocessing)

[//]: # (    steps which were imported from`my_library.utils.preprocessing` file and converts preprocessed data into a ' filename ' + `_preprocessed`csv file)

[//]: # (    - <a href = 'https://github.com/BLEND360/UVyper/blob/idc_dev1/notebooks/kmeans.ipynb'>`k_means.ipynb`</a> works on the preprocessed data and performs kmeans clustering.)

[//]: # (    - <a href = 'https://github.com/BLEND360/UVyper/blob/idc_dev1/notebooks/hierarchical.ipynb'>`hierarchical.ipynb`</a> works on the preprocessed data and performs hierarchical clustering.)

[//]: # (    - <a href = 'https://github.com/BLEND360/UVyper/blob/idc_dev1/notebooks/gmm.ipynb'>`gmm.ipynb`</a> works on the preprocessed data and performs gmm clustering.)

[//]: # (    - <a href = 'https://github.com/BLEND360/UVyper/blob/idc_dev1/notebooks/birch.ipynb'>`birch.ipynb`</a> works on the preprocessed data and performs birch clustering.)

[//]: # (    - <a href = 'https://github.com/BLEND360/UVyper/blob/idc_dev1/notebooks/dbscan.ipynb'>`dbscan.ipynb`</a> works on the preprocessed data and performs dbscan clustering.)

- `UVyper.py` is the main file which contains the implementation of all the clustering algorithms.
- `Uvyper.ipynb` is the jupyter notebook which imports `UVyper.py` file and performs all the clustering algorithms on the preprocessed data.