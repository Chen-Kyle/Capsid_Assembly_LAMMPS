#!/bin/bash

# This script parses over values of Eatt
#
# This produces the following files for each Eatt:
#   ${SLURM_JOB_ID}/decamer.lammps                    atom/bond topology
#   ${SLURM_JOB_ID}/gaussian_native_{A,B,C,D}.table   tabulated Gaussian potentials
#   ${SLURM_JOB_ID}/forces.lammps                     pair/bond style commands (mode-dependent)
#   ${SLURM_JOB_ID}/harmonic_bond_coeffs.lammps       bond_coeff lines for ENM bonds
#   ${SLURM_JOB_ID}/native_contact_pair_coeffs.lammps (--type hybrid only)
#
# Importantly the trajectory file is:
#   ${SCRATCH}${SLURM_JOB_ID}/seg.dcd

seed_vals=(42)

# PDB files (adjust paths if yours differ)
PDB="important_oligomer_pdbs/cg_ABCD_avg.pdb" #"important_oligomer_pdbs/cg_ABCD_separate.pdb"

# Output directory
# output_dir="${SCRATCH}HBV_enm/${SLURM_JOB_ID}" (set in run_lammps.sh)

# Enative values to sweep (1.0 was the original value: it appears reasonable)
Enative_vals=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5)

# Simulation length in timesteps (10 fs each, so 1000000 = 10 ns)
nsteps=1000000


# Output directory
output_tag="Enative"
output_dir_top_level="${SCRATCH}HBV_enm/${output_tag}/${SLURM_JOB_ID}"

# ---------------------------------------------------------------------------
# Main loop: for each Enative, run nseed independent simulations
# ---------------------------------------------------------------------------

for seed in "${seed_vals[@]}"
do
    for Enative in "${Enative_vals[@]}"
    do
        sbatch run_lammps.sh ${seed} ${PDB} ${Enative} ${nsteps} ${output_top_level}
    done
done