#!/bin/sh
export BRiQ_BINPATH=/home/asia/Asia/Software/RNA-BRiQ/build/bin 
export BRiQ_DATAPATH=/home/asia/Asia/Software/RNA-BRiQ/BRiQ_data 

INPUT=input     # Input file 
OUTPDB=pred.pdb   # Output: refined structure
RANDOMSEED=0   # Random seed  

## Generate an initial PDB structure from the given sequence  
$BRiQ_BINPATH/BRiQ_InitPDB GGGUCAGGCCGGCGAAAGUCGCCACAGUUUGGGGAAAGCUGUGCAGCCUGUAACCCCCCCACGAAAGUGGG init.pdb  
## Predict RNA structure        
$BRiQ_BINPATH/BRiQ_Predict $INPUT $OUTPDB $RANDOMSEED
