# Haplotype_imputation
A major goal in population genetics is to predict the genetic history of contemporary populations from sequence data (1) (2). In experimental and agricultural genetics there are many cases where multiple founders (of known genotype) are combined to produce recombinant progeny (3) and for each descendant, reconstructing its haplotype means finding which regions along the genome descend from which founder. 
In the last years, geneticists started to unfold the power of Convolutional Neural Networks in population genetic inference (5). Our goal is to explore the potential and robustness of Neural Networks for haplotype reconstruction, and secondarily for predicting numbers of recombination events. We worked on simulations before applying our model to the C. elegans multiparental experimental evolution (CeMEE) dataset (6), with the Single Nucleotide Polymorphisms (SNPs) of the chromosome I as a case study.

As a L3 biology student at the Ecole Normale Supérieure of Paris (ENS), I worked under the direction of Luke Noble of the team Experimental Evolutionary Genetics of the ENS. The training last from November 2018 to May 2019 and our goal was to explore the potential and robustness of Neural Networks for haplotype reconstruction, and secondarily for predicting numbers of recombination events.

This repository contains the code for neural networks able to reconstruct haplotypes from genotypes, implemented in Python 3.6.7 with keras 2.2.4 based on tensorflow 1.13.1.

**A presentation of the problem and the principal architectures are presented in the main notebook "Haplotype_imputation_Artififial_NN.ipynb"**, which gives you the links to the other notebooks describing the different networks, when needed. For example, the successively tested CNN architectures are in "CNN_models.ipynb".

All required data is provided :
- "Initial_data" : Single Nucleotide Polymorphisms (SNPs) from the CeMEE dataset and the genetic map of *Caenorhabditis elegans*, note that one file has to be unzipped
- Genotypes and haplotypes I constructed myself : "Simulations_..." (recombined simulated founders) and "Recombination_..." (recombined real founders)
- "Linkage_..." : Formatted genetic maps for the first chromosome of *C. elegans* and selected numbers of SNPs

Finally, **we evaluated our best model accuracy for varying genotypes similarities between founders (both mean and variance) and we placed in this map classical datasets in population genetics**. You can find this in the notebook "Accuracy_similarity" into the "Similarity" folder.

References :
(1) Crow and Kimura. An Introduction to Population Genetics Theory, 1970 (Harper & Row)
(2) Wakeley. Coalescent Theory: an introduction, 2009 (Roberts & Company Pub.)
(3) Rakshit S., Rakshit A. & Patil. Journal of Genetics, 2012 (91: 111.)
(5) Flagel, Brandvain, Schrider. Molecular Biology and Evolution, 2019 (vol. 36, p. 220–238)
(6) Noble, Chelo et al. Genetics, 2017 (vol. 207 no. 4 1663-1685)
