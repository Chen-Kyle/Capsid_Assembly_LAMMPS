#!/bin/bash
#SBATCH --account=hagan-lab
#SBATCH --partition=hagan-compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --output=slurm_%j.out


# HBV Decamer CG Oligomer Simulation
#
# Command line arguements in order are:
#         1. Seed number
#         2. PDB file path
#         3. Enative value
#         4. Number of timesteps
#
# This script runs both:
#         generate_lammps_data.py
#         decamer.lammps
#
# This produces:
#   ${SLURM_JOB_ID}/decamer.lammps                    atom/bond topology
#   ${SLURM_JOB_ID}/gaussian_native_{A,B,C,D}.table   tabulated Gaussian potentials
#   ${SLURM_JOB_ID}/forces.lammps                     pair/bond style commands (mode-dependent)
#   ${SLURM_JOB_ID}/harmonic_bond_coeffs.lammps       bond_coeff lines for ENM bonds
#   ${SLURM_JOB_ID}/native_contact_coeffs.lammps      (default mode only)
#   ${SLURM_JOB_ID}/native_contacts.pairs             (--type pairs only)
#   ${SLURM_JOB_ID}/native_contact_pair_coeffs.lammps (--type hybrid only)
#
# Importantly the trajectory file is:
#   ${SCRATCH}${SLURM_JOB_ID}/seg.dcd

#conda activate westpa2
# Number of seeds passed as first argument (default 5)
seed=${1:-42}

# PDB files
pdb_default="${HBV_ENM_PATH}/scripts/important_oligomer_pdbs/abcd_capsid.pdb" #"important_oligomer_pdbs/cg_ABCD_separate.pdb"
PDB=${2:-${pdb_default}}

# Enative (default value = 1.0)
Enative=${3:-1.0}

# Simulation length in timesteps (10 fs each timestep)
nsteps=${4:-100000000}

# Output directory
output_dir_toplevel=${5:-"${SCRATCH}/HBV_enm/Enative=${Enative}_seed=${seed}"}
use_job_id=${6:-"yes"}
if [ "${use_job_id}" = "yes" ]; then
    output_dir="${output_dir_toplevel}/${SLURM_JOB_ID}"
else
    output_dir="${output_dir_toplevel}"
fi
# ---------------------------------------------------------------------------
# Logs simulation data
# ---------------------------------------------------------------------------

echo -e "\n$(date) JOBID:${SLURM_JOB_ID}     Enative:${Enative}     PDB File:${PDB}     Seed Num:${seed}     Output Directory:${output_dir}\n" >> master.log


# ---------------------------------------------------------------------------
# Generates the LAMMPS files
# ---------------------------------------------------------------------------

# Generate shared input files for this Enative value.
# --type hybrid uses standard pair_style table — compatible with 2018 module.
echo "$(printf '%0.s-' {1..100})"
echo -e "\npython generate_lammps_data.py --Enative ${Enative} --pdb ${PDB} --output_dir ${output_dir}\n"
echo "$(printf '%0.s-' {1..100})"

python generate_lammps_data.py  \
    --pdb    "${PDB}"           \
    --output_dir "${output_dir}"\
    --Enative "${Enative}"      \


# ---------------------------------------------------------------------------
# Runs the LAMMPS simulation
# ---------------------------------------------------------------------------

echo -e "Finished running generate_lammps_data.py\n"
echo "$(printf '%0.s-' {1..100})"
echo -e "\nmpirun -n $SLURM_NTASKS lmp -in lammps_oligomer.in -var myseed ${seed} -var nsteps ${nsteps} -var output_dir ${output_dir}\n"
echo "$(printf '%0.s-' {1..100})"

# The default value for large systems is 8 but for <1000 atoms it should be 1
time mpirun -n $SLURM_NTASKS lmp -in lammps_oligomer.in  \
    -var output_dir ${output_dir}            \
    -var nsteps ${nsteps}                    \
    -var myseed ${seed}                      \

# Starts running detect_clusters on the trajectory file
echo "Running sbatch run_detect_clusters.sh on ${output_dir}" 
sbatch run_detect_clusters.sh ${PDB} ${output_dir}/seg.dcd 
