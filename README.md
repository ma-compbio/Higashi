# Multiscale and integrative single-cell Hi-C analysis with Higashi

This is the implementation of the computational framework Higashi for scHi-C analysis.
<img src="https://github.com/ma-compbio/Higashi/blob/main/figs/logo.png" align="right"
     alt="logo" width="240">

Higashi has four main components.
1.  Represent the scHi-C dataset as a hypergraph, where each cell and each genomic bin are represented as cell node and genomic bin node, respectively.  Each non-zero entry in the single-cell contact map is modeled as a hyperedge connecting the corresponding cell and the two genomic loci of that particular chromatin interaction. The read count for each chromatin interaction is used as attribute of the hyperedge. (Figure a)
2. We train a hypergraph neural network based on the constructed hypergraph or unveiling high-order interaction patterns. (Figure b)
3.  We extract the embedding vectors of cell nodes from the trained hypergraph neural network for downstream analysis.
4.  We then use the trained hypergraph neural network to impute single-cell Hi-C contact maps with the flexibility to incorporate the latent correlations between cells to enhance overall imputation, enabling detailed characterization of 3D genome features such as TAD-like domain boundaries and A/B compartment scores at single-cell resolution.

We have developed new computational strategies that can reliably compare A/B compartment scores and TAD-like domain boundaries across individual cells which will also be included in this repo.

![figs/Overview.png](https://github.com/ma-compbio/Higashi/blob/main/figs/Overview.png)


# Requirements
The core part of Higashi (`Process.py`, `main_cell.py`)

- h5py
- numpy
- pytorch (tested on 1.4.0)
- scikit-learn
- tqdm

Generating visualization plots
- seaborn
- matplotlib
- UMAP

It is known that under pytorch 1.7.0, there will be a "backward error" (required FloatTensor but received DoubleTensor.) We are inspecting the cause of the error. The library will be upgraded to support the latest pytorch.

## Input format
The input files include:
1. `data.txt`, a tab separated file with the following columns: `['cell_name','cell_id', 'chrom1', 'pos1', 'chrom2', 'pos2', 'count']` (We will support the SCool format in the future. Detailed documentaion of the SCOOL format can be found at https://cooler.readthedocs.io/en/latest/schema.html?highlight=scool#single-cell-single-resolution)
2. (optional) `label_info.pickle`, a python pickle file of a dictionary storing labeled information of cells. The structure of the dictionary:
  
  ```
  {
    'cell type': ['GM12878', 'K562', 'NHEK',.....,'GM12878'],
    'coverage':[12000, 14000, ...., 15000],
    'batch':['batch_1', 'batch_1',..., 'batch_2']
  }
  ``` 
   The order of the labeled vector should be consistent with 'cell_id' of the `data.txt`
  
 3. (optional) `sc_signal.hdf5`, a hdf5 file for storing the coassayed signals. The structure of the hdf5 file:
 
 ```
 ./"signal1"
  ./"bin" (If it's not something based on the genomic bin, then leave it out)
    ./"chrom"
    ./"start"
    ./"end"
  ./"0"
  ./"1"
  ./"2"
./"signal2"
...

```


 

## Configure the parameters
All customizable parameters are stored in a JSON config file. An example config file can be found in `config_dir/example.JSON`
| params       | description                  | example                   |
|--------------|------------------------------|---------------------------|
| config_name| name of this configuration, will be used in visualization tool |"sn-m3C-seq-with_meth"
|  data_dir| directory where the data are stored | "/work/magroup/ruochiz/Data/scHiC_collection/sn-m3C-seq"
|  temp_dir| directory where the temporary files will be stored | "../Temp/sn-m3C_1Mb"
|  genome_reference_path| path of the genome reference file from USCS  Genome Browser, will be usde to generate bin nodes | "../hg19.chrom.sizes.txt"
|  cytoband_path| path of the cytoband reference file from USCS Genome Browser, will be used to remove centromere regions | "../cytoBand_hg19.txt"
|  chrom_list| list of chromosomes to train the model on | ["chr1", "chr2","chr3","chr4","chr5"]
|  resolution| resolution for training and imputation | 1000000
|  resolution_cell| resolution for generate attributes of the cell nodes | 1000000,
|  minimum_distance| minimum genomic distance between a pair of genomc bins to impute | 1000000,
|  maximum_distance|  maximum genomic distance between a pair of genomc bins to impute (-1 represents no constraint)| -1
|  local_transfer_range| number of neighboring bins in 1D genomic distance to consider during imputation (similar to the window size of linear convolution) | 1
|  dimensions| embedding dimensions | 64,
|  impute_list| list of chromosome to impute (must appear in the chrom list above)|["chr1"]
|  neighbor_num| number of neighboring cells to consider | 5
|  embedding_name| name of embedding vectors to store | "exp1"
|  coassay| using co-assayed signals or not | true
|  coassay_signal| name of the co-assayed signals in the hdf5 file to use (can be empy) | "meth_cg-100kb-cg_rate"


## Usage
### Run the Higashi main program
#### Commands
1. `cd Code`
2. Run `python Process.py -c {CONFIG_PATH}`. Run the data processing pipeline for the given configuration file
3. Run `python main_cell.py -c {CONFIG_PATH}`. Train the Higashi model for the given configuration file
#### Output
Embedding vectors and imputed matrix at the `temp_dir` of the given configuration file

### Calling and calibrating single cell TAD-like domain boundaries
Under construction
### Calling single cell compartment scores
Under construction
### One more thing
Under construction


## Cite

If you want to cite our paper


