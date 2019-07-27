# Haplotype_imputation
Contains the code for neural networks able to reconstruct haplotypes from genotypes, implemented in Python 3.6.7 with keras 2.2.4 based on tensorflow 1.13.1.

A presentation of the problem and the principal architectures are presented in the main notebook "Haplotype_imputation_Artififial_NN.ipynb". More specifically, the successively tested CNN architectures are in "CNN_models.ipynb".

All recuire data are provided, both :
- "Initial_data" : Single Nucleotide Polymorphisms (SNPs) from the CeMEE dataset and the genetic map of Caenorhabditis elegans
- Genotypes and haplotypes I constructed myself : "Simulations_..." (recombined simulated founders) and "Recombination_..." (recombined real founders)