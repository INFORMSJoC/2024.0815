#!/bin/sh

module load stack/2022.2
module load python
module load py-numpy
module load py-pandas
module load py-matplotlib
module load py-scikit-learn
module load py-torch

for MYSEED in $(seq 2023 2122)
do

    sed "s/MYSEED/${MYSEED}/g" hpc_many_fixed_seed.sh > tmp${MYSEED}.sh
    chmod +x tmp${MYSEED}.sh
    qsub tmp${MYSEED}.sh
    rm -rf tmp*

done
