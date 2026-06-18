#!/bin/bash

# HBV Decamer CG Oligomer Simulation
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

# conda activate westpa2
# Number of seeds passed as first argument (default 5)
nseed=1

PDB="important_oligomer_pdbs/abcd_capsid.pdb" #"important_oligomer_pdbs/cg_ABCD_separate.pdb"
output_dir="lammps_out"
Enative_vals=(1.0)
nsteps=1000000

# ---------------------------------------------------------------------------
# Main loop: for each Enative, run nseed independent simulations
# ---------------------------------------------------------------------------

# Generates the LAMMPS files
echo "$(printf '%0.s-' {1..100})"
echo -e "\npython generate_lammps_data.py --pdb ${PDB} --output_dir ${output_dir}\n"
echo "$(printf '%0.s-' {1..100})"
python generate_lammps_data.py  \
    --pdb    "${PDB}"           \
    --output_dir "${output_dir}"\

# Runs the LAMMPS simulation
echo -e "Finished running generate_lammps_data.py\n"
echo "$(printf '%0.s-' {1..100})"
echo -e "\nmpirun -n 8 lmp -in lammps_oligomer.in -var output_dir ${output_dir} -var nsteps ${nsteps}\n"
echo "$(printf '%0.s-' {1..100})"
time lmp -in lammps_oligomer.in   \
    -var output_dir ${output_dir} \
    -var nsteps ${nsteps}         \

echo "All runs complete."
