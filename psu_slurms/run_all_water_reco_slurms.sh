for file in /storage/group/dfc13/default/dcolson/reco_*.slurm; do
    sbatch "$file"
done