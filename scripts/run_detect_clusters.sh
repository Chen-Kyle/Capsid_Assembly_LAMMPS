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
pdb_default="${HBV_ENM_PATH}/scripts/important_oligomer_pdbs/abcd_capsid.pdb"
PDB=${1:-${pdb_default}}

traj_default="/scratch0/kylechen/HBV_enm/Enative=1.0_seed=42/10738499/seg.dcd"
traj_file=${2:-${traj_default}}


output_dir="${HBV_ENM_PATH}/scripts/lammps_out"
contacts=20

# ---------------------------------------------------------------------------
# Main loop: for each Enative, run nseed independent simulations
# ---------------------------------------------------------------------------

# Logs starting the analysis
echo "$(printf '%0.s-' {1..100})"
echo -e "\npython full_traj_analysis.py --pdb ${PDB} --traj ${traj_file} --contacts ${contacts}\n"
echo "$(printf '%0.s-' {1..100})"
time python ${HBV_ENM_PATH}/scripts/full_traj_analysis.py  \
    --pdb        "${PDB}"       \
    --traj       "${traj_file}" \
    --contacts   "${contacts}"  \

# Runs the LAMMPS simulation
echo -e "Finished running detect_clusters.py\n"
echo "$(printf '%0.s-' {1..100})"
