#!/bin/sh
RNA="5TPY"
filepath="./"
C=0.01
gamma=0.1
sd_cut=3.7
random_seed=123
#$BRiQ_path=
#source activate base  
python CODA2_scoring.py $RNA $filepath $C $gamma $sd_cut

python CODA2_MC.py $RNA $filepath $random_seed

sec=$(head -n 1 2D.dot)
seq=$(sed -n '2p' data/$RNA.fasta)
length=${#seq}
nwc=$(printf "%.0s." $(seq 1 $length))

mkdir 3D/
cd 3D/
cat > input <<EOF
pdb init.pdb
seq $seq
sec $sec
nwc $nwc
EOF

cat >run_briq.sh <<EOF
#!/bin/sh
## Change \$BRiQ_BINPATH and \$BRiQ_DATAPATH to the directories containing the
## binary executables and precompiled data files correspondingly. 
export BRiQ_BINPATH=$BRiQ_path/RNA-BRiQ-main/build/bin 
export BRiQ_DATAPATH=$BRiQ_path/BRiQ_data 

INPUT=input     # Input file 
OUTPDB=$pred.pdb   # Output: refined structure
RANDOMSEED=$random_seed   # Random seed  

## Generate an initial PDB structure from the given sequence  
\$BRiQ_BINPATH/BRiQ_InitPDB $seq init.pdb  
## Predict RNA structure        
\$BRiQ_BINPATH/BRiQ_Predict \$INPUT \$OUTPDB \$RANDOMSEED
EOF

#bash run_briq.sh

cd ../
