from setuptools import setup, find_packages

setup(
    name='higashi',
    version='0.2.0',
    description='Higashi: Multiscale and integrative scHi-C analysis',
    url='https://github.com/ma-compbio/Higashi',
    include_package_data=True,
    packages=find_packages(exclude=['Temp', 'config_dir', 'figs']),
    install_requires=[
        'python>=3.7.0',
        'h5py',
        'numpy',
        'pandas',
        'pytorch>=1.7.0',
        'fbpca',
        'scikit-learn>=0.23.2',
        'tqdm',
        'cooler>=0.8',
        'seaborn',
        'matplotlib',
        'umap-learn',
        'bokeh>=2.1.1',
        'Pillow'
    ],
    author='Ruochi Zhang',
    author_email='ruochiz@andrew.cmu.edu',
    license='MIT'
)