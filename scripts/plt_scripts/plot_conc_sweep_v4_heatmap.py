"""
plot_conc_sweep_v4_heatmap.py

Loads every complete_cluster_data.pkl found nested under a given sweep
directory, e.g.:
    Enative_concentration_sweep_v4/Enative=0.2_seed=42/10742082/complete_cluster_data.pkl

For each trajectory, the JOBID is looked up in master.log to recover the
lattice PDB that was actually simulated (Enative=*_seed=* alone does not
encode box size -- different JOBIDs under the same Enative/seed folder can
come from different lattice=..._blength=....pdb files), and the box side
length ("blength", Angstroms) is parsed from that PDB's filename.

The largest cluster size at the trajectory's final frame is then computed
and plotted as a heatmap: side_length (x) vs Enative (y), colored by max
cluster size. When multiple trajectories land in the same (Enative,
side_length) cell (replicate seeds/jobs), their values are averaged.
"""

import os
import re
import glob
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

HBV_ENM_PATH = os.environ.get("HBV_ENM_PATH", "/home/kyle/2026_Research/HBV_enm")

SWEEP_DIR_DEFAULT = "/home/kyle/storage/kyle_storage/HBV_enm/trajectory_files/Enative_concentration_sweep_v4"
MASTER_LOG        = f"{HBV_ENM_PATH}/scripts/master.log"
FIGURES_DIR       = f"{HBV_ENM_PATH}/raw_data"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pkl_dir', default=SWEEP_DIR_DEFAULT,
                    help="Top-level sweep directory containing "
                         "Enative=*_seed=*/JOBID/complete_cluster_data.pkl")
    p.add_argument('--master_log', default=MASTER_LOG,
                    help="master.log used to look up the PDB simulated for each JOBID")
    p.add_argument('--cluster_key', default='all_well_formed_clusters',
                    choices=['all_well_formed_clusters', 'all_clusters'],
                    help="Which cluster set (from the pkl) to compute max cluster size from")
    p.add_argument('--output_dir', default=FIGURES_DIR,
                    help="Directory to save the output figure to")
    p.add_argument('--output_name', default='conc_sweep_v4_max_cluster_heatmap.png',
                    help="Output figure filename")
    return p.parse_args()


def parse_master_log(log_file):
    """
    Returns dict: job_id (str) -> PDB path (str), as recorded in master.log.
    """
    job_to_pdb = {}
    pattern = re.compile(r'JOBID:(\d+).*?PDB File:(\S+)')
    with open(log_file) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                job_to_pdb[m.group(1)] = m.group(2)
    return job_to_pdb


def pkl_enative(pkl_path):
    """
    Extract enative: float from a pkl path, or None.
    """
    m = re.search(r'Enative=([0-9.]+)', pkl_path)
    return float(m.group(1)) if m else None


def pdb_side_length(pdb_path):
    """
    Extract side_length (blength, Angstroms): int from a lattice PDB path,
    or None if this PDB isn't a lattice=..._blength=....pdb file.
    """
    m = re.search(r'blength=([0-9]+)', os.path.basename(pdb_path))
    return int(m.group(1)) if m else None


def max_cluster_size_last_frame(pkl_path, cluster_key):
    """
    Loads clusters[cluster_key] from pkl_path and returns the largest
    cluster size (number of segments) at the trajectory's final frame.
    """
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)
    clusters = d[cluster_key]
    last_frame = max(clusters.keys())
    sizes = [len(c['segments']) for c in clusters[last_frame]]
    return max(sizes) if sizes else 0


def make_heatmap(matrix, enatives, side_lengths, title, out_path):
    width  = max(1.3 * len(side_lengths) + 1.5, 8)
    height = 0.85 * len(enatives) + 1.5
    fig, ax = plt.subplots(figsize=(width, height))

    masked = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(color='0.85')

    im = ax.imshow(masked, cmap=cmap, origin='lower', aspect='auto')

    ax.set_xticks(range(len(side_lengths)))
    ax.set_xticklabels(side_lengths, rotation=45, ha='right', fontsize=12)
    ax.set_yticks(range(len(enatives)))
    ax.set_yticklabels(enatives, fontsize=12)

    ax.set_xlabel('Side length (blength, Å)', fontsize=15)
    ax.set_ylabel('Binding Energy', fontsize=15)
    ax.set_title(title, fontsize=16)

    for i in range(len(enatives)):
        for j in range(len(side_lengths)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                        color='white' if val < np.nanmax(matrix) * 0.6 else 'black',
                        fontsize=11)

    cbar = fig.colorbar(im, ax=ax, label='max cluster size (last frame)')
    cbar.set_label('max cluster size (last frame)', fontsize=13)
    cbar.ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    args = parse_args()

    job_to_pdb = parse_master_log(args.master_log)
    print(f"Loaded {len(job_to_pdb)} job->PDB mappings from {args.master_log}\n")

    pkl_files = glob.glob(f"{args.pkl_dir}/**/complete_cluster_data.pkl", recursive=True)
    print(f"Found {len(pkl_files)} pkl files under {args.pkl_dir}\n")

    # (enative, side_length) -> list of max cluster sizes
    cell_values = {}

    for pkl_path in sorted(pkl_files):
        enative = pkl_enative(pkl_path)
        if enative is None:
            print(f"[skip] can't parse Enative: {pkl_path}")
            continue
        if enative == 2.0:
            print(f"[skip] excluding Enative=2.0: {pkl_path}")
            continue

        job_id = os.path.basename(os.path.dirname(pkl_path))
        pdb_path = job_to_pdb.get(job_id)
        if pdb_path is None:
            print(f"[skip] job {job_id} not found in master.log: {pkl_path}")
            continue

        side_length = pdb_side_length(pdb_path)
        if side_length is None:
            print(f"[skip] PDB has no blength (not a lattice PDB): {pdb_path}")
            continue

        try:
            max_size = max_cluster_size_last_frame(pkl_path, args.cluster_key)
        except Exception as e:
            print(f"[skip] error reading {pkl_path}: {e}")
            continue

        print(f"  Enative={enative}  side_length={side_length}  job={job_id}  max_cluster_size={max_size}")
        cell_values.setdefault((enative, side_length), []).append(max_size)

    if not cell_values:
        raise SystemExit("No trajectories with a resolvable Enative + side_length were found.")

    enatives     = sorted({k[0] for k in cell_values})
    side_lengths = sorted({k[1] for k in cell_values})

    matrix = np.full((len(enatives), len(side_lengths)), np.nan)
    for (enative, side_length), values in cell_values.items():
        i = enatives.index(enative)
        j = side_lengths.index(side_length)
        if len(values) > 1:
            print(f"  averaging {len(values)} replicates for Enative={enative} side_length={side_length}")
        matrix[i, j] = np.mean(values)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = f"{args.output_dir}/{args.output_name}"
    make_heatmap(matrix, enatives, side_lengths,
                title='Max Cluster Size at Final Frame (Binding Energy vs Side Length)',
                out_path=out_path)

    print("Done.")
