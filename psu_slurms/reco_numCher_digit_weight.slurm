#!/bin/bash 

#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=50G
#SBATCH --array=91
#SBATCH --job-name=numCher_digit_weight_recos

module purge
module load anaconda

conda activate dc_hitman10

### Change to the working directory where your code is located
echo $PWD
### Call your executable
# python3 /storage/group/dfc13/default/dcolson/HITMAN/hitman/all_sensor/numCher_digit_weight_reco.py -i *_${SLURM_ARRAY_TASK_ID}.ntuple.root -n /storage/group/dfc13/default/dcolson/BNL/data/infile_data4/network_numCher_digit_weight_all -o /storage/group/dfc13/default/dcolson/BNL/data/infile_data4/recos/numCher_digit_weight_all/reco_numCher_digit_weight_all_${SLURM_ARRAY_TASK_ID}.pkl --print_numpy True
DATE="07_29_25"
LOC="disk_d_phase_3"
TYPE="weight_wBP"

# Define file paths
INPUT_FILE="/storage/group/dfc13/default/dcolson/my_pickles/${DATE}_${LOC}_${TYPE}.pkl"
NETWORK_FILE="/storage/group/dfc13/default/dcolson/BNL/data/infile_data4/network_numCher_digit_weight_all"
OUTPUT_FILE="/storage/group/dfc13/default/dcolson/BNL/data/infile_data4/recos/numCher_digit_weight_all/\
${DATE}_${LOC}_${TYPE}_reco_numCher_digit_weight_all.pkl"

# Run the Python script
python3 /storage/group/dfc13/default/dcolson/HITMAN/hitman/all_sensor/real_data_numCher_digit_weight_reco.py\
    -i "$INPUT_FILE" \
    -n "$NETWORK_FILE" \
    -o "$OUTPUT_FILE" \
    --print_numpy True
