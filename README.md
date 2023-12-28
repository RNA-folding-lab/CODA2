# CODA2
# Introduction
This pipeline is used to determine the 3D structures from deep mutational sequencing data for RNAs.

# Requirement
python 3
scikit-learn
CODA (https://github.com/zh3zh/CODA)
RNA-BRiQ (https://github.com/Jian-Zhan/RNA-BRiQ)

# Usage
1. input files (put them in folder of data/):
   1.1. RNA_name.var.ra: this is the fitness file for your RNA (e.g.,RNA_name), which can be generated from deep mutational sequencing data using the CODA (the previous version of CODA2, see: https://github.com/zh3zh/CODA).
   1.2. RNA_name.fasta: the sequence of your RNA.
2. change the run.sh file:
   RNA="RNA_name"
   BRiQ_path="the BRiQ path in your computer"
4. bash run.sh

# Outputs:
1. score/: CODA score matrix, Score_map
2. MC/: mc_energy.txt, contact map with lowest energy, predicted secondary structure (dot-bracket and txt format)
3. 3D/: predicted 3D structure by BRiQ (pred.pdb)

# Reference
1. Zhang, Z., Xiong, P., Zhang, T., Wang, J., Zhan, J., & Zhou, Y. (2020). Accurate inference of the full base-pairing structure of RNA by deep mutational scanning and covariation-induced deviation of activity. Nucleic Acids Research, 48(3), 1451–1465. https://doi.org/10.1093/nar/gkz1192
2. Xiong, P., Wu, R., Zhan, J., & Zhou, Y. (2021). Pairing a high-resolution statistical potential with a nucleobase-centric sampling algorithm for improving RNA model refinement. Nature communications, 12(1), 2777. https://doi.org/10.1038/s41467-021-23100-4
