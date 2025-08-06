#!/bin/bash 

#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=50G
#SBATCH --array=91
#SBATCH --job-name=elosstank_digit_edge_recos

module purge
module load anaconda
conda activate dc_hitman10

### Change to the working directory where your code is located
echo $PWD
### Call your executable
# python3 /storage/group/dfc13/default/dcolson/HITMAN/hitman/all_sensor/elosstank_digit_edge_reco.py -i *_${SLURM_ARRAY_TASK_ID}.ntuple.root -n /storage/group/dfc13/default/dcolson/BNL/data/infile_data4/network_elosstank_digit_edge_all -o /storage/group/dfc13/default/dcolson/BNL/data/infile_data4/recos/elosstank_digit_edge_all/reco_elosstank_digit_edge_all_${SLURM_ARRAY_TASK_ID}.pkl --print_numpy True
# python3 /storage/group/dfc13/default/dcolson/HITMAN/hitman/all_sensor/real_data_elosstank_digit_edge_reco.py -i /storage/group/dfc13/default/dcolson/my_pickles/07_09_25_processed_data_disk_e_phase_3_CFD_1.pkl -n /storage/group/dfc13/default/dcolson/BNL/data/infile_data4/network_elosstank_digit_edge_all -o /storage/group/dfc13/default/dcolson/BNL/data/infile_data4/recos/elosstank_digit_edge_all/07_09_25_reco_elosstank_digit_edge_all_disk_e_phase_3_1.pkl --print_numpy True # --event_limit 1000
DATE="07_27_25"
LOC="disk_l_phase_9"
TYPE="cfd_1000"

INPUT_FILE="/storage/group/dfc13/default/dcolson/my_pickles/${DATE}_${LOC}_${TYPE}.pkl"
NETWORK_FILE="/storage/group/dfc13/default/dcolson/wbls_sim/network_elosstank_digit_edge_all"
OUTPUT_FILE="/storage/group/dfc13/default/dcolson/wbls_sim/recos/elosstank_digit_edge_all/\
${DATE}_${LOC}_${TYPE}_reco_elosstank_digit_edge_all.pkl"

# Run the Python script
python3 /storage/group/dfc13/default/dcolson/HITMAN/hitman/all_sensor/real_data_elosstank_digit_edge_reco.py\
    -i "$INPUT_FILE" \
    -n "$NETWORK_FILE" \
    -o "$OUTPUT_FILE" \
    --print_numpy True
