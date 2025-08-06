#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:2
#SBATCH --constraint=v100nv|v100s
#SBATCH --partition=mgc-mri
#SBATCH --account=dfc13_mri
#SBATCH --time=24:00:00

module purge
module load anaconda
conda activate dc_hitman10

python3 /storage/group/dfc13/default/dcolson/HITMAN/hitman/all_sensor/wbls_train_hitman_numCher_digit_edge.py -i /storage/group/dfc13/default/dcolson/wbls_sim/*_{1..90}.ntuple.root -o /storage/group/dfc13/default/dcolson/wbls_sim/network_numCher_digit_edge_all --save_history --epochs 200
