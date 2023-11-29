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
