#!/bin/bash
#SBATCH --account=hagan-lab
#SBATCH --partition=hagan-compute-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_%j.out


# HBV Decamer CG Oligomer Analysis Batch Cluster Script
#
# Recursively finds every seg.dcd nested under a given top-level directory
# (e.g. /scratch0/kylechen/HBV_enm/Enative_concentration_sweep_v4) and runs
# full_traj_analysis.py on each one, e.g.:
#   Enative_concentration_sweep_v4/Enative=0.2_seed=42/10742082/seg.dcd
#
# The PDB used for each trajectory is looked up in master.log by JOBID
# (the seg.dcd's parent directory name), NOT a single fixed PDB for the
# whole sweep -- different JOBIDs under the same Enative=X_seed=Y folder
# can come from different lattice PDBs (different box/side lengths), same
# as batch_run_lammps.sh's PDB x Enative loop. The PDB_fallback argument is
# only used when a JOBID can't be found in master.log.
#
# This produces, alongside each seg.dcd:
#   ${traj_dir_path}/complete_cluster_data.pkl   pkl file containing cluster and bond data
#
# NOTE: unlike run_detect_clusters.sh (one trajectory per job), this script
# processes every trajectory found sequentially within a single job. If the
# sweep directory has many trajectories, this may exceed the time limit of
# the hagan-compute-short partition -- switch to hagan-compute if needed.

# Gets command line arguments
pdb_fallback_default="${HBV_ENM_PATH}/scripts/important_oligomer_pdbs/abcd_capsid.pdb"
PDB_fallback=${1:-${pdb_fallback_default}}

sweep_dir_default="/scratch0/kylechen/HBV_enm/Enative_concentration_sweep_v4"
sweep_dir=${2:-${sweep_dir_default}}

master_log_default="${HBV_ENM_PATH}/scripts/master.log"
master_log=${3:-${master_log_default}}

contacts=20

# ---------------------------------------------------------------------------
# Looks up the PDB used for a given JOBID in master.log, e.g.:
#   JOBID:10742082 ... PDB File:/work/kylechen/HBV_enm/scripts/lattice_pdbs/lattice=cubic_Ndimers=60_blength=1000.pdb ...
# and re-roots it under this machine's ${HBV_ENM_PATH}/scripts/ (everything
# after "/scripts/" in the remote path is assumed to match the local layout).
# Echoes the local path, or nothing if the JOBID/PDB entry can't be resolved.
# ---------------------------------------------------------------------------
resolve_pdb_for_job() {
    local job_id="$1"
    local pdb_remote
    pdb_remote=$(grep -oP "JOBID:${job_id}\b.*?PDB(?: File)?:\K\S+" "${master_log}" | tail -1)
    if [[ -z "${pdb_remote}" ]]; then
        return 1
    fi

    local pdb_suffix="${pdb_remote#*/scripts/}"
    if [[ "${pdb_suffix}" == "${pdb_remote}" ]]; then
        # "/scripts/" not found in the remote path -- can't safely re-root it
        return 1
    fi

    local pdb_local="${HBV_ENM_PATH}/scripts/${pdb_suffix}"
    if [[ ! -f "${pdb_local}" ]]; then
        return 1
    fi

    echo "${pdb_local}"
}

# ---------------------------------------------------------------------------
# Main loop: find every seg.dcd nested under sweep_dir and analyze it
# ---------------------------------------------------------------------------

mapfile -t traj_files < <(find "${sweep_dir}" -type f -name "seg.dcd" | sort)

echo "$(printf '%0.s-' {1..100})"
echo -e "\nFound ${#traj_files[@]} seg.dcd files under ${sweep_dir}\n"
echo "$(printf '%0.s-' {1..100})"

for traj_file in "${traj_files[@]}"; do
    job_id="$(basename "$(dirname "${traj_file}")")"

    PDB="$(resolve_pdb_for_job "${job_id}")"
    if [[ -z "${PDB}" ]]; then
        echo "[warn] could not resolve PDB for job ${job_id} from ${master_log} -- falling back to ${PDB_fallback}"
        PDB="${PDB_fallback}"
    fi

    echo "$(printf '%0.s-' {1..100})"
    echo -e "\npython full_traj_analysis.py --pdb ${PDB} --traj ${traj_file} --contacts ${contacts}\n"
    echo "$(printf '%0.s-' {1..100})"
    time python ${HBV_ENM_PATH}/scripts/full_traj_analysis.py  \
        --pdb        "${PDB}"       \
        --traj       "${traj_file}" \
        --contacts   "${contacts}"  \

    echo -e "Finished running detect_clusters.py on ${traj_file}\n"
done

echo "$(printf '%0.s-' {1..100})"
echo -e "Finished batch analysis of ${#traj_files[@]} trajectories\n"
echo "$(printf '%0.s-' {1..100})"
