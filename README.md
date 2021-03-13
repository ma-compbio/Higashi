
# Higashi: Multiscale and integrative scHi-C analysis
<img src="https://github.com/ma-compbio/Higashi/blob/main/figs/logo2.png" align="right"
     alt="logo" width="290">

https://doi.org/10.1101/2020.12.13.422537

As a computational framework for scHi-C analysis, Higashi has the following features:

-  Higashi represents the scHi-C dataset as a **hypergraph** (Figure a) 
     - Each cell and each genomic bin are represented as the cell node and the genomic bin node.
     - Each non-zero entry in the single-cell contact map is modeled as a hyperedge. 
     - The read count for each chromatin interaction is used as the attribute of the hyperedge. 
- Higashi uses a **hypergraph neural network** to unveil high-order interaction patterns within this constructed hypergraph. (Figure b)
- Higashi can produce the **embeddings** for the scHi-C for downstream analysis.
-  Higashi can **impute single-cell Hi-C contact maps**, enabling detailed characterization of 3D genome features such as **TAD-like domain boundaries** and **A/B compartment scores** at single-cell resolution.

--------------------------

![figs/Overview.png](https://github.com/ma-compbio/Higashi/blob/main/figs/short_overview.png)



# Documentation
Please see [the wiki](https://github.com/ma-compbio/Higashi/wiki) for extensive documentation and example tutorials.

Higashi is constantly being updated, see [change log](https://github.com/ma-compbio/Higashi/wiki/Change-Log) for the updating history

# Cite

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



# Contact

Please contact ruochiz@andrew.cmu.edu or raise an issue in the github repo with any questions about installation or usage. 
