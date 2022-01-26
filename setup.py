from setuptools import setup, find_packages
print (find_packages())
setup(
    name='higashi',
    version='0.1.0a0',
    description='Higashi: Multiscale and integrative scHi-C analysis',
    url='https://github.com/ma-compbio/Higashi',
    include_package_data=True,
    python_requires='>=3.9',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.2',
        'scipy==1.7.3',
        'pandas==1.3.4',
        'cython>=0.29.24',
        'torch>=1.8.0',
        'fbpca',
        'scikit-learn>=0.23.2',
        'tqdm',
        'h5py',
        'cooler>=0.8',
        'seaborn>=0.11.2',
        'umap-learn>=0.5',
        'bokeh>=2.1.1',
        'Pillow'
    ],
    extras_require={},
    author='Ruochi Zhang',
    author_email='ruochiz@andrew.cmu.edu',
    license='MIT'
)

