#!/bin/bash

# This script parses over values of Eatt and different pdb files
#
# This produces the following files for each combination of Eatt and pdb:
#   ${SLURM_JOB_ID}/decamer.lammps                    atom/bond topology
#   ${SLURM_JOB_ID}/gaussian_native_{A,B,C,D}.table   tabulated Gaussian potentials
#   ${SLURM_JOB_ID}/forces.lammps                     pair/bond style commands (mode-dependent)
#   ${SLURM_JOB_ID}/harmonic_bond_coeffs.lammps       bond_coeff lines for ENM bonds
#
# Importantly the trajectory file is:
#   ${SCRATCH}${SLURM_JOB_ID}/seg.dcd

seed_vals=(42)

# PDB files (adjust paths if yours differ)
#PDB="${HBV_ENM_PATH}/scripts/important_oligomer_pdbs/cg_ABCD_avg.pdb" #"important_oligomer_pdbs/cg_ABCD_separate.pdb"
PDB_dir="${HBV_ENM_PATH}/scripts/lattice_pdbs"

# Output directory
# output_dir="${SCRATCH}HBV_enm/${SLURM_JOB_ID}" (set in run_lammps.sh)
output_tag="Enative_concentration_sweep"

# Enative values to sweep (1.0 was the original value: it appears reasonable)
Enative_vals=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5)

# Simulation length in timesteps (10 fs each, so 1000000 = 10 ns)
nsteps=1000000


# ---------------------------------------------------------------------------
# Main loop: for each Enative, run nseed independent simulations
# ---------------------------------------------------------------------------

echo "----------Starting new batch run----------" >> master.log

for seed in "${seed_vals[@]}"
do  
    for file in ${PDB_dir}/*;
    do
        PDB=$file
        for Enative in "${Enative_vals[@]}"
        do
            output_dir_toplevel="${SCRATCH}/HBV_enm/${output_tag}/Enative=${Enative}_seed=${seed}"
            sbatch run_lammps.sh ${seed} ${PDB} ${Enative} ${nsteps} ${output_dir_toplevel}
        done
    done
done