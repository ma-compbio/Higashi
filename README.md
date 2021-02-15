
# Higashi
<img src="https://github.com/ma-compbio/Higashi/blob/main/figs/logo2.png" align="right"
     alt="logo" width="290">

[***Multiscale and integrative single-cell Hi-C analysis with Higashi***](https://www.biorxiv.org/content/10.1101/2020.12.13.422537v1)

As a computational framework for scHi-C analysis, Higashi has the following features:

-  Higashi represents the scHi-C dataset as a **hypergraph** (Figure a) 
     - Each cell and each genomic bin are represented as the cell node and the genomic bin node.
     - Each non-zero entry in the single-cell contact map is modeled as a hyperedge. 
     - The read count for each chromatin interaction is used as the attribute of the hyperedge. 
- Higashi uses a **hypergraph neural network** to unveil high-order interaction patterns within this constructed hypergraph. (Figure b)
- Higashi can produce the **embeddings** for the scHi-C for downstream analysis.
-  Higashi can **impute single-cell Hi-C contact maps**, enabling detailed characterization of 3D genome features such as **TAD-like domain boundaries** and **A/B compartment scores** at single-cell resolution.

--------------------------

![figs/Overview.png](https://github.com/ma-compbio/Higashi/blob/main/figs/Overview.png)


## Requirements
**Running Higashi**
- Python (>=3.5.0, tested on 3.7.9)
- h5py (tested on 2.10.0)
- numpy (tested on 1.19.2)
- pandas (tested on 1.1.3)
- pytorch (tested on 1.4.0)
- fbpca (tested on 1.0.0)
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


## Change Log

### 2021-2-14
#### Features
- We now use fbpca to handle PCA of extremely large feature matrices
- Beta version of removing batch effects of scHi-C (by including batch_id as part of the input)
- Memory usage optimization (The memory usage is now 20% of the previous version on the sn-m3c-seq dataset)
- Remove the optional smoothing and quantile normalization options due to computational efficiency
- Allow customizable UMAP/TSNE parameters for Higashi-vis
- Include linear-conv+rwr imputation results for visualization

[History change log](https://github.com/ma-compbio/Higashi/blob/main/Changelog.md)


## How to use
### Step 1: Prepare the input files
All these input files should be put under the same directory. The path to this directory will be needed in the next step.
1. `data.txt`, a tab separated file with the following columns: `['cell_name','cell_id', 'chrom1', 'pos1', 'chrom2', 'pos2', 'count']` (We will support the SCool format in the future. Detailed documentaion of the SCool format can be found at https://cooler.readthedocs.io/en/latest/schema.html?highlight=scool#single-cell-single-resolution)
2. `label_info.pickle`, a python pickle file of a dictionary storing labeled information of cells. If there is no labeled information, please create an empty dictionary in python and save it as a pickle. An example of the structure of the dictionary and how to save it as the pickle:
  
  ```python
  import pickle
  output_label_file = open("label_info.pickle", "wb")
  label_info = {
    'cell type': ['GM12878', 'K562', 'NHEK',.....,'GM12878'],
    'coverage':[12000, 14000, ...., 15000],
    'batch':['batch_1', 'batch_1',..., 'batch_2'],
    ...
  }
  pickle.dump(label_info, output_label_file)
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


### Step 2: Configure the running parameters
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
|  batch_id  | str | Optional. The name of the batch id information stored in `label_info.pickle`. The corresponding information would be used to remove batch effects | "batch id"

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

#### Visualization related parameters
| params       |Type | description                  | example                   |
|--------------|-----|------------------------------|---------------------------|
| UMAP_params | dict | Parameters that'll be passed to Higashi-vis. Higashi-vis will use these parameters when calculating UMAP visualization. Follow the naming convention of the package umap | {"n_neighbors": 30, "min_dist": 0.3|
|TSNE_params | dict | Similar to UMAP_params. Follow the naming convention of tsne in sklearn | {"n_neighbors": 15}
| random_walk | bool | Whether run linear_convolution and randomwalk-with-restart at the processing part for visualization. Code adapted from scHiCluster. Do not recommend when resolution goes higher than 100Kb. | false    
### Step 3: Data processing 
Run the following commands to process the input data.
```bash
cd Code
python Process.py -c {CONFIG_PATH}
```
Fill in the `{CONFIG_PATH}` with the path to the configuration JSON file that you created in the step 2. This script will finish the following tasks:
- generate a dictionary that'll map genomic bin loci to the node id.
- extract data from the data.txt and turn that into the format of hyperedges (triplets)
- create contact maps based on sparse scHi-C for visualization, baseline model, and generate node attributes
- run linear convolution + random-walk-with-restart (scHiCluster) to impute the contact maps as baseline and visualization
- **(Optional)** smooth the contact maps 
- **(Optional)** quantile normalization
- generate node attributes
- **(Optional)** process co-assayed signals

Before each step is executed, a message would be printed indicating the progress, which helps the debugging process.

### Step 4: Train the Higashi model
```bash
python main_cell.py -c {CONFIG_PATH} -s {START_STEP}
```
Fill in the `{CONFIG_PATH}` with the path to the configuration JSON file that you created in the step 2.
Fill in the `{START_STEP}` with 1,2,3 which indicates the following steps:
1. Train Higashi **without** cell-dependent GNN to force self-attention layers to capture the heterogeneity of chromatin structures
2. Train Higashi **with** cell-dependent GNN, but with **k=0**
3. Train Higashi **wit**h cell-dependent GNN, but with **k=`k`** in the configuration JSON
When `{START_STEP}` is 1, the program would execute step 1,2,3 sequentially. 

#### Output
The output is stored at the `temp_dir`, which contains
1. Embedding vectors for the given configuration file. The embedding vectors are saved with name `{embedding_name}\_{id}\_origin.npy`. The `{embedding_name}` is the parameter in the configuration file. The `{id}` starts at 0, ends with the number of chromosomes that is contained in the training data. 0 corresponds to the cell embeddings. 1 ~ corresponds to embeddings of bins from each chromosome.
2. Imputed matrices. The imputed matrices are saved with name `{chr1}_{embedding_name}_nbr_{k}_impute.hdf5`. The format of the imputed matrix is an hdf5 file with the structure
 ```
 .
 ├── coordinates (vector size of k x 2)
 ├── cell 0 (vector size of k)
 ├── cell 1
 ├── ...
 └── cell N
```
The matrix can be generated by putting the vector of `cell *` to the corresponding entries of the `coordinates`.
We also provide a script that can transform this output hdf5 file to the SCool format. Detailed documentaion of the SCool format can be found at https://cooler.readthedocs.io/en/latest/schema.html?highlight=scool#single-cell-single-resolution). To do that run the following command

```bash
python Higashi2SCool.py -c {CONFIG_PATH} -o {OUTPUT_PATH} -n {NEIGHBOR_FLAG}
```
Fill in `{CONFIG_PATH}` with the path of the configuration file, `{OUTPUT_PATH}` with the path of the output. `{NEIGHBOR_FLAG}` stands for using neighboring cells in the embedding space to help imputation or not, when set as `false` only the imputation result of Higashi(0) would be transformed into the Scool format. When set as `true`, the imputation results of Higashi with the given `k` in the configuration file would be transformed into the Scool format.

### Step 5: Compartment / TAD-like structure calling
In progress

--------------------

## One more thing (visualization of scHi-C & Higashi analysis)
<img src="https://github.com/ma-compbio/Higashi/blob/main/figs/logo_vis2.png" align="right"
     alt="logo" width="290">
     
In Higashi, we also implemented a **visualization tool** which helps the users enjoy Higashi better. The visualization tool allows interactive navigation of the scHi-C analysis results and is implemented based on [bokeh](https://docs.bokeh.org/en/latest/index.html).
     
To launch the visualization tool, first create a `visual_config.JSON` file under the `config_dir`. 
The directory structure should be
```
 .
 ├── Higashi
 │   ├── Code
 │   └── config_dir
 └──     └── visual_config.JSON
```
The `visual_config.JSON` file has the following content:
```JSON
{
  "config_list": [
    {PATH1},
    {PATH2},
  ]
}
```
After that, just run the following commands
```bash
cd Code
bokeh serve --port={PORT} --address=0.0.0.0 --allow-websocket-origin=*:{PORT} Higashi_vis/
```
Finally, open a browser and go to `{IP}:{PORT}/Higashi_vis`. If you are running the program with a PC, the `{IP}` can just be localhost. If you are running this on a server, `{IP}` would be the ip address of the server. 
Notably, the server itself should allow accession of this `{PORT}`.
If you see the following interface, you have successfully launched the visualization tool.
![figs/screen.png](https://github.com/ma-compbio/Higashi/blob/main/figs/screen1.png)

A detailed tutorial on the functions of this visualization tool can be found at [Here](https://github.com/ma-compbio/Higashi/tree/main/Code/Higashi_vis) (still in progress)


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

