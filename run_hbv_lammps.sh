#!/bin/bash
#SBATCH --account=hagan-lab
#SBATCH --partition=hagan-compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module purge
source /work/kylechen/miniconda3/etc/profile.d/conda.sh
conda activate westpa2

LAMMPS_BIN="/work/kylechen/installations/lammps-22Jul2025/build/lmp"
MPIRUN="/work/kylechen/envs/westpa2/bin/mpirun"

# Number of seeds passed as first argument (default 5)
nseed=1
# Root directory containing the PDB, contact files, and scripts
SIMDIR="${SLURM_SUBMIT_DIR}"

# PDB files (adjust paths if yours differ)
PDB="${SIMDIR}/important_oligomer_pdbs/cg_ABCD_separate.pdb"
BOUND="${SIMDIR}/important_oligomer_pdbs/cg_ABCD_avg.pdb"

# Enative values to sweep
Enative_vals=(1.0)

# Simulation length in timesteps (10 fs each, so 1000000 = 10 ns)
NSTEPS=1000000

# Repulsion scale (dimensionless)
EREPULSION=1.0

# ---------------------------------------------------------------------------
# Main loop: for each Enative, run nseed independent simulations
# ---------------------------------------------------------------------------
for enative in "${Enative_vals[@]}"; do

    # Generate shared input files for this Enative value.
    # --type hybrid uses standard pair_style table — compatible with 2018 module.
    echo "Generating LAMMPS input files for Enative=${enative}..."
    cd "${SIMDIR}"
    python3 generate_lammps_data.py \
        --pdb   "${PDB}"           \
        --bound "${BOUND}"         \
        --conndir  connect_files   \
        --contactdir .             \
        --Enative  ${enative}      \
        --type pairs

    for (( i=1; i<=${nseed}; i++ )); do

        run_dir="${SIMDIR}/${enative}/${i}"
        mkdir -p "${run_dir}"

        # Copy all input files into the run directory so each run is isolated
        # and output files (seg.dcd, log.lammps, etc.) do not collide.
        cp decamer.lammps                   "${run_dir}/"
        cp harmonic_bond_coeffs.lammps      "${run_dir}/"
        cp native_contact_pair_coeffs.lammps "${run_dir}/"
        cp forces.lammps                    "${run_dir}/"
        cp gaussian_native_*.table          "${run_dir}/"
        cp lammps_oligomer.in               "${run_dir}/"

        echo "  Running seed ${i} in ${run_dir}..."
        cd "${run_dir}"

        "${MPIRUN}" "${LAMMPS_BIN}" \
            -in  lammps_oligomer.in         \
            -var Erepulsion ${EREPULSION}   \
            -var myseed     ${i}            \
            -var nsteps     ${NSTEPS}

        cd "${SIMDIR}"
    done

done

echo "All runs complete."
