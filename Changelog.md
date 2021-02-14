# Change Log

## 2021-2-14
### Features
- We now use fbpca to handle PCA of extremely large feature matrices
- Beta version of removing batch effects of scHi-C (by including batch_id as part of the input)
- Memory usage optimization (The memory usage is now 20% of the previous version on the sn-m3c-seq dataset)
- Remove the optional smmoth and quantile normalization option due to computational efficiency
