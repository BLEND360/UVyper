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
pip install kneed
pip install yellowbrick==1.5
pip install scikit-learn==0.24.1 
pip install plotly
pip install kaleido==0.1.0.post1
``` 

#### Sample data used : <a href='https://github.com/BLEND360/UVyper/blob/idc_dev1/notebooks/cvs_hcb_member_profiling.csv'>`cvs_hcb_member_profiling.csv`</a>

#### <a href = 'https://github.com/BLEND360/UVyper/tree/idc_dev1/notebooks'>`notebooks`</a> folder contains:

- <a href = 'https://github.com/BLEND360/UVyper/blob/idc_dev1/notebooks/UVyper.py'>`UVyper.py`</a> is the main file
  which contains the implementation of all the clustering algorithms.
- <a href = 'https://github.com/BLEND360/UVyper/blob/idc_dev1/notebooks/UVyper.ipynb'>`Uvyper.ipynb`</a> is the jupyter
  notebook which imports `UVyper.py` file and performs all the clustering algorithms on the preprocessed data.
