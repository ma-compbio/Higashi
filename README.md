
# Higashi
<img src="https://github.com/ma-compbio/Higashi/blob/main/figs/logo2.png" align="right"
     alt="logo" width="300">

[***Multiscale and integrative single-cell Hi-C analysis with Higashi***](https://www.biorxiv.org/content/10.1101/2020.12.13.422537v1)

As a computational framework for scHi-C analysis, Higashi has the following features:

-  Higashi represents the scHi-C dataset as a **hypergraph** (Figure a) 
     - Each cell and each genomic bin are represented as cell node and genomic bin node.
     - Each non-zero entry in the single-cell contact map is modeled as a hyperedge. 
     - The read count for each chromatin interaction is used as attribute of the hyperedge. 
- Higashi uses a **hypergraph neural network** to unveil high-order interaction patterns within this constructed hypergraph. (Figure b)
- Higashi can produce the **embeddings** for the scHi-C for downstream analysis.
-  Higashi can **impute single-cell Hi-C contact maps** , enabling detailed characterization of 3D genome features such as **TAD-like domain boundaries** and **A/B compartment scores** at single-cell resolution.

--------------------------

![figs/Overview.png](https://github.com/ma-compbio/Higashi/blob/main/figs/Overview.png)


## Requirements
**Running Higashi**
- Python (>=3.5.0, tested on 3.7.9)
- h5py (tested on 2.10.0)
- numpy (tested on 1.19.2)
- pandas (tested on 1.1.3)
- pytorch (tested on 1.4.0)
- scikit-learn (tested on 0.23.2)
- tqdm (tested on 4.50.2)

**Generating visualization plots**
- seaborn
- matplotlib
- UMAP

**Interactive visualization sessions**
- bokeh (tested on 2.1.1)
- PIL (tested on 7.2.0)
- cachetools (tested on 4.1.1)

**Note**: It is known that under pytorch 1.7.0, there will be a "backward error" (required FloatTensor but received DoubleTensor.) We are inspecting the cause of the error. The library will be upgraded to support the latest pytorch.




## How to use
### Step 1: Prepare the input files
All these input files should be put under the same directory. The path to this directory will be needed in the next step.
1. `data.txt`, a tab separated file with the following columns: `['cell_name','cell_id', 'chrom1', 'pos1', 'chrom2', 'pos2', 'count']` (We will support the SCool format in the future. Detailed documentaion of the SCool format can be found at https://cooler.readthedocs.io/en/latest/schema.html?highlight=scool#single-cell-single-resolution)
2. **(Optional)** `label_info.pickle`, a python pickle file of a dictionary storing labeled information of cells. An example of the structure of the dictionary:
  
  ```
  {
    'cell type': ['GM12878', 'K562', 'NHEK',.....,'GM12878'],
    'coverage':[12000, 14000, ...., 15000],
    'batch':['batch_1', 'batch_1',..., 'batch_2'],
    ...
  }
  ``` 
   The order of the labeled vector should be consistent with the `'cell_id'` column of the `data.txt`
  
 3. **(Optional)** `sc_signal.hdf5`, a hdf5 file for storing the coassayed signals. The structure of the hdf5 file:
 
 ```
 .
 ├── signal1
 │   ├── bin (info mation about entries in the signal 1 file. If the signal 1 is not based on genomic coordinates, left this option out)
 │   │   ├── chrom
 │   │   ├── start
 │   │   └── end
 │   ├── 0 (signals, size should be the same as signal1/bin/chrom)
 │   ├── 1
 │   └── 2
 └── signal2
 │   ├── ...
 └──
```


### Step 2: configure the running parameters
All customizable parameters are stored in a JSON config file. An example config file can be found in `config_dir/example.JSON`. The path to this JSON config file will be needed in Step 3.

#### Input data related parameters

| params       |Type | description                  | example                   |
|--------------|-----|------------------------------|---------------------------|
| config_name| str| Name of this configuration, will be used in visualization tool |"sn-m3C-seq-with_meth"
|  data_dir| str| Directory where the data are stored | "/sn-m3C-seq"
|  temp_dir| str| Directory where the temporary files will be stored. An empty folder will be created if it doesn't exists. | "../Temp/sn-m3C_1Mb"
|  genome_reference_path| str| Path of the genome reference file from USCS  Genome Browser, will be usde to generate bin nodes | "../hg19.chrom.sizes.txt"
|  cytoband_path| str|Path of the cytoband reference file from USCS Genome Browser, will be used to remove centromere regions | "../cytoBand_hg19.txt"
|  coassay| bool | Using co-assayed signals or not | true
|  coassay_signal| str| Name of the co-assayed signals in the hdf5 file to use (can be empy) | "meth_cg-100kb-cg_rate"
|  optional_smooth|bool |Smooth when calculating features for cell nodes |false
| optional_quantile|bool |Quantile normalization when calculating feautures for cell nodes | false

#### Training process related parameters
| params       |Type | description                  | example                   |
|--------------|-----|------------------------------|---------------------------|
|  chrom_list| str| List of chromosomes to train the model on. The name convention should be the same as the data.txt and the genome_reference file | ["chr1", "chr2","chr3","chr4","chr5"]
|  resolution|int| Resolution for training and imputation. | 1000000
|  resolution_cell|int| Resolution for generate attributes of the cell nodes. Recommend to use 1Mb or 500Kb for computational efficiency. | 1000000
|  local_transfer_range| int| Number of neighboring bins in 1D genomic distance to consider during imputation (similar to the window size of linear convolution) | 1
|  dimensions|int| Embedding dimensions | 64,
| loss_mode | str |Train the model in classification or ranking (can be either classification or rank) | rank
| rank_thres |  int | Difference of ground truth values that are larger than rank_thres would be considered as stable order.| 1


#### Output related parameters
| params       |Type | description                  | example                   |
|--------------|-----|------------------------------|---------------------------|
|  embedding_name| str | Name of embedding vectors to store | "exp1"
|  impute_list| int| List of chromosome to impute (must appear in the chrom list above)|["chr1"]
|  minimum_distance|int| Minimum genomic distance between a pair of genomc bins to impute (bp) | 1000000
|  maximum_distance|int|  Maximum genomic distance between a pair of genomc bins to impute (bp, -1 represents no constraint)| -1
|  neighbor_num| int| Number of neighboring cells to incorporate when making imputation | 5

#### Computational resources related parameters
| params       |Type | description                  | example                   |
|--------------|-----|------------------------------|---------------------------|
|  cpu_num | int| Higashi is optimized for multiprocessing. Limit the number of cores to use with this param. -1 represents use all available cpu.  |-1
|  gpu_num | int| Higashi is optimized to utilize multiple gpus for computational efficiency. Higashi won't use all these gpus throughout the time. For co-assayed data, it would use multiple gpus in the processing step. For all data, Higashi would train and impute scHi-C on different gpus for computational efficiency. This parameters should be non negative. |8


### Step 3: Enjoy Higashi
#### Commands
1. `cd Code`
2. Run `python Process.py -c {CONFIG_PATH}`. Run the data processing pipeline for the given configuration file.
3. Run `python main_cell.py -c {CONFIG_PATH}`. Train the Higashi model for the given configuration file.

#### Output
The output is stored at the `temp_dir`, which contains
1. Embedding vectors for the given configuration file. The embedding vectors are saved with name {embedding_name}\_{id}\_origin.npy. The {embedding_name} is the parameter in the configuration file. The {id} starts at 0, end with the number of chromosomes that is contained in the training data. 0 corresponds to the cell embeddings. 1 ~ corresponds to embeddings of bins from each chromosome.
2. Imputed matrices. The imputed matrciesare saved with name . The format of the imputed matrix is an hdf5 file with the structure
 ```
 .
 ├── coordinates (vector size of k x 2)
 ├── cell 0 (vector size of k)
 ├── cell 1
 ├── ...
 └── cell N
```
The matrix can be generated by putting the vector of `cell *` to the corresponding entries of the `coordinates`.

#### Calling and calibrating single cell TAD-like domain boundaries
Under construction
#### Calling single cell compartment scores
Under construction

#### One more thing
Under construction


## Cite

Cite our paper by

```
@article {Zhang2020multiscale,
	author = {Zhang, Ruochi and Zhou, Tianming and Ma, Jian},
	title = {Multiscale and integrative single-cell Hi-C analysis with Higashi},
	year = {2020},
	doi = {10.1101/2020.12.13.422537},
	publisher = {Cold Spring Harbor Laboratory},
	journal = {bioRxiv}
}
```

![figs/Overview.png](https://github.com/ma-compbio/Higashi/blob/main/figs/paper.png)

