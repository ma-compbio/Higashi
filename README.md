
# Higashi: Multiscale and integrative scHi-C analysis
<img src="https://github.com/ma-compbio/Higashi/blob/main/figs/logo2.png" align="right"
     alt="logo" width="290">

https://doi.org/10.1038/s41587-021-01034-y

As a computational framework for scHi-C analysis, Higashi has the following features:

-  Higashi represents the scHi-C dataset as a **hypergraph**
     - Each cell and each genomic bin are represented as the cell node and the genomic bin node.
     - Each non-zero entry in the single-cell contact map is modeled as a hyperedge. 
     - The read count for each chromatin interaction is used as the attribute of the hyperedge. 
- Higashi uses a **hypergraph neural network** to unveil high-order interaction patterns within this constructed hypergraph.
- Higashi can produce the **embeddings** for the scHi-C for downstream analysis.
-  Higashi can **impute single-cell Hi-C contact maps**, enabling detailed characterization of 3D genome features such as **TAD-like domain boundaries** and **A/B compartment scores** at single-cell resolution.

--------------------------

![figs/Overview.png](https://github.com/ma-compbio/Higashi/blob/main/figs/short_overview.png)

# Installation

We now have Higashi on conda.

`conda install -c ruochiz higashi`

It is recommended to have pytorch installed (with CUDA support when applicable) before installing higashi.

# Documentation
Please see [the wiki](https://github.com/ma-compbio/Higashi/wiki) for extensive documentation and example tutorials.

Higashi is constantly being updated, see [change log](https://github.com/ma-compbio/Higashi/wiki/Change-Log) for the updating history

# Tutorial
- [4DN sci-Hi-C (Kim et al.)](https://github.com/ma-compbio/Higashi/blob/main/tutorials/4DN_sci-Hi-C_Kim%20et%20al.ipynb)
- [Ramani et al.](https://github.com/ma-compbio/Higashi/blob/main/tutorials/Ramani%20et%20al.ipynb)

# Cite

Cite our paper by

```
@article {Zhang2020multiscale,
	author = {Zhang, Ruochi and Zhou, Tianming and Ma, Jian},
	title = {Multiscale and integrative single-cell Hi-C analysis with Higashi},
	year={2021},
	publisher = {Nature Publishing Group},
	journal = {Nature biotechnology}
}
```

![figs/Overview.png](https://github.com/ma-compbio/Higashi/blob/main/figs/higashi_title.png)



# Contact

Please contact ruochiz@andrew.cmu.edu or raise an issue in the github repo with any questions about installation or usage. 
