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

module load anaconda
conda activate /storage/group/dfc13/default/abald/envs/hitman_env
python3 /storage/group/dfc13/default/abald/HITMAN/hitman/per_sensor/train_hitman_numCher_digit_weight_persensor.py -i *_{1..90}.ntuple.root -o network_numCher_digit_weight_per --save_history --epochs 200
