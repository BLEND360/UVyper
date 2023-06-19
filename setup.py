from setuptools import setup, find_packages

setup(
    name='UVyper',
    version='1.0.0',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    url='https://github.com/BLEND360/UVyper/tree/idc_dev1',
    author='Blend360',
    description='UVyper is a python package for unsupervised learning',
    install_requires=[
        'matplotlib',
        'statsmodels',
        'ortools',
        'yellowbrick==1.5',
        'plotly',
        'kaleido==0.1.0.post1',
        'scikit-learn-extra',
        'k-means-constrained'
    ]
)
