#!/bin/bash -l
#PBS -l nodes=1:ppn=24:gpus=2,walltime=2:00:00 
#PBS -N FastMRI_train
#PBS -q v100-8
#PBS -e err_fastmri_train
#PBS -o output_fastmri_train

cd /home/csci5980/liang664/fastMRI
module load conda
source activate fastmri
module load cuda cuda-sdk
module load python3/3.8.3_anaconda2020.07_mamba

export LD_LIBRARY_PATH=/home/siepmann/liang664/.conda/envs/fastmri/lib:$LD_LIBRARY_PATH

python train.py

