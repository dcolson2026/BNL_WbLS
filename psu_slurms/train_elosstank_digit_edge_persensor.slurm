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
# conda activate /storage/group/dfc13/default/abald/envs/hitman_env

python3 /storage/group/dfc13/default/dcolson/HITMAN/hitman/per_sensor/wbls_train_hitman_elosstank_digit_edge_persensor.py -i /storage/group/dfc13/default/dcolson/wbls_sim/*_{1..90}.ntuple.root -o /storage/group/dfc13/default/dcolson/wbls_sim/network_elosstank_digit_edge_per --save_history --epochs 200
