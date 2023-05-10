## Install the below-mentioned packages after installing the latest updated Vyper model
### Refer to the page 5 of below documentation link to install the latest updated Vyper
<a href="https://github.com/BLEND360/UVyper/blob/idc_dev1/DS%20AS9%20Vyper%20Installation%20Manual.pdf">Vyper
Installation Manual</a>
### After installing the latest updated Vyper, install the below-mentioned packages
``` 
pip install lightgbm
pip install matplotlib
pip install varclushi
pip install k-means-constrained==0.6.0
pip install statsmodels --upgrade
pip install ortools==9.3.10459
import ortools
import ortools.graph.pywrapgraph
pip install kneed
pip install yellowbrick==1.5
pip install scikit-learn==0.24.1 
pip install plotly
``` 

#### Sample data used : <a href='https://github.com/BLEND360/UVyper/blob/idc_dev1/notebooks/cvs_hcb_member_profiling.csv'>`cvs_hcb_member_profiling.csv`</a>

[//]: # (#### <a href = 'https://github.com/BLEND360/UVyper/tree/idc_dev1/my_library'>`my_library`</a> folder contains:)

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

#### <a href = 'https://github.com/BLEND360/UVyper/tree/idc_dev1/notebooks'>`notebooks`</a> folder contains:
- <a href = 'https://github.com/BLEND360/UVyper/blob/idc_dev1/notebooks/UVyper.py'>`UVyper.py`</a> is the main file which contains the implementation of all the clustering algorithms.
- <a href = 'https://github.com/BLEND360/UVyper/blob/idc_dev1/notebooks/UVyper.ipynb'>`Uvyper.ipynb`</a> is the jupyter notebook which imports `UVyper.py` file and performs all the clustering algorithms on the preprocessed data.
