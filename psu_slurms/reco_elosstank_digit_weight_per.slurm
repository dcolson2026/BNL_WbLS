#!/bin/bash 

#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=50G
#SBATCH --array=91
#SBATCH --job-name=tankdigit_weightper_recos

module purge
module load anaconda

conda activate dc_hitman10

### Change to the working directory where your code is located
echo $PWD
### Call your executable
# python3 /storage/group/dfc13/default/dcolson/HITMAN/hitman/per_sensor/elosstank_digit_weight_per_reco.py -i *_${SLURM_ARRAY_TASK_ID}.ntuple.root -n /storage/group/dfc13/default/dcolson/BNL/data/infile_data4/network_elosstank_digit_weight_per -o /storage/group/dfc13/default/dcolson/BNL/data/infile_data4/recos/elosstank_digit_weight_per/elosstank_digit_weight_per_reco_${SLURM_ARRAY_TASK_ID}.pkl --print_numpy True
# python3 /storage/group/dfc13/default/dcolson/HITMAN/hitman/per_sensor/real_data_elosstank_digit_weight_per_reco.py -i /storage/group/dfc13/default/dcolson/my_pickles/07_09_25_processed_data_disk_e_phase_3_Weight_1.pkl -n /storage/group/dfc13/default/dcolson/BNL/data/infile_data4/network_elosstank_digit_weight_per -o /storage/group/dfc13/default/dcolson/BNL/data/infile_data4/recos/elosstank_digit_weight_per/07_09_25_reco_elosstank_digit_weight_per_disk_e_phase_3_1.pkl --print_numpy True # --event_limit 1000
DATE="07_29_25"
LOC="disk_d_phase_3"
TYPE="weight_wBP"

INPUT_FILE="/storage/group/dfc13/default/dcolson/my_pickles/${DATE}_${LOC}_${TYPE}.pkl"
NETWORK_FILE="/storage/group/dfc13/default/dcolson/BNL/data/infile_data4/network_elosstank_digit_weight_per"
OUTPUT_FILE="/storage/group/dfc13/default/dcolson/BNL/data/infile_data4/recos/elosstank_digit_weight_per/\
${DATE}_${LOC}_${TYPE}_reco_elosstank_digit_weight_per.pkl"

# Run the Python script
python3 /storage/group/dfc13/default/dcolson/HITMAN/hitman/per_sensor/real_data_elosstank_digit_weight_per_reco.py\
    -i "$INPUT_FILE" \
    -n "$NETWORK_FILE" \
    -o "$OUTPUT_FILE" \
    --print_numpy True
