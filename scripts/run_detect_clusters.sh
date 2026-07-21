#!/bin/bash
#SBATCH --account=hagan-lab
#SBATCH --partition=hagan-compute-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_%j.out


# HBV Decamer CG Oligomer Analysis Cluster Script
# 
# Runs detect_clusters.py given the cmd line arguments
# 
# This produces:
#   ${traj_dir_path}/cluster_data.pkl         pkl file containing cluster and bond data

# Gets command line arguments
PDB="${1:?Usage: $0 <pdb_file>}"
traj_file="${2:?Usage: $0 <traj_file>}"


output_dir="${HBV_ENM_PATH}/scripts/lammps_out"
contacts=10
cutoff=8.0 # Normally doesn't do anything since detect_clusters is using computed cutoffs from claude
           # Won't do anything unless u change it in the detect_clusters.py script manually

# ---------------------------------------------------------------------------
# Main section for starting to run the cluster analysis on the trajectory
# ---------------------------------------------------------------------------

# Logs starting the analysis
echo "$(printf '%0.s-' {1..100})"
echo -e "\npython detect_clusters.py --pdb ${PDB} --traj ${traj_file} --cutoff ${cutoff} --contacts ${contacts}\n"
echo "$(printf '%0.s-' {1..100})"
python ${HBV_ENM_PATH}/scripts/detect_clusters.py  \
    --pdb        "${PDB}"       \
    --traj       "${traj_file}"      \
    --cutoff     "${cutoff}"    \
    --contacts   "${contacts}"  \

# Runs the LAMMPS simulation
echo -e "Finished running detect_clusters.py\n"
echo "$(printf '%0.s-' {1..100})"