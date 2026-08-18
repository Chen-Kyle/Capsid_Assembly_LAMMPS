"""
plot_cluster_panels.py
Loads all cluster pkl files produced by batch_detect_clusters.py and plots:

  Figure 1: largest cluster size vs time     -> panel_max_cluster_size.png
  Figure 2: number of clusters (size >= 2)   -> panel_num_clusters.png

Each figure is a grid with blength on the y-axis (rows) and Enative on the
x-axis (columns).  Rows and columns are inferred from the pkl files found.
"""

import os
import re
import glob
import pickle
import argparse
import matplotlib.pyplot as plt

HBV_ENM_PATH = os.environ.get("HBV_ENM_PATH", "/home/kyle/2026_Research/HBV_enm")

OUTPUT_DIR  = f"{HBV_ENM_PATH}/raw_data/cluster_data"
MASTER_LOG  = f"{HBV_ENM_PATH}/scripts/master.log"
FIGURES_DIR = f"{HBV_ENM_PATH}/raw_data"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pkl_dir',
                   help="Directory of the pickle files to plot", default=f'{HBV_ENM_PATH}/raw_data/pkl_files/Enative_concentration_sweep_v2')
    p.add_argument('--output_dir',
                   help="Directory to send the panel plot to")
    p.add_argument('')



def parse_master_log(log_file):
    """
    Returns dict: job_id (str) -> blength (int or None).
    """
    job_to_blength = {}
    pattern = re.compile(r'JOBID:(\d+).*?PDB File:(\S+)')
    with open(log_file) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                job_id   = m.group(1)
                pdb_path = m.group(2)
                bl       = re.search(r'blength=(\d+)', pdb_path)
                job_to_blength[job_id] = int(bl.group(1)) if bl else None
    return job_to_blength


def pkl_metadata(pkl_path):
    """
    Extract (enative: float, job_id: str) from a pkl path, or (None, None).
    """
    en = re.search(r'Enative=([0-9.]+)_seed=\d+', pkl_path)
    jb = re.search(r'Enative=[^/]+/(\d+)/', pkl_path)
    if not en or not jb:
        return None, None
    return float(en.group(1)), jb.group(1)


def compute_stats(clusters):
    """
    Returns (frames, max_sizes, num_clusters) sorted by frame.
      max_sizes:    largest cluster size per frame
      num_clusters: count of clusters with >= 2 members per frame
    """
    frames = sorted(clusters.keys())
    max_sizes    = []
    num_clusters = []
    for f in frames:
        sizes = [len(c['segments']) for c in clusters[f]]
        max_sizes.append(max(sizes) if sizes else 0)
        num_clusters.append(sum(1 for s in sizes if s >= 2))
    return frames, max_sizes, num_clusters


def make_panel_figure(data, blengths, enatives, ylabel, title, out_path):
    """
    data: dict[(enative, blength)] -> (frames, values)
    blengths: sorted list of row labels
    enatives: sorted list of column labels
    """
    nrows, ncols = len(blengths), len(enatives)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(2.8 * ncols, 2.4 * nrows),
                             sharex=False, sharey=False)

    if nrows == 1:
        axes = [axes]
    if ncols == 1:
        axes = [[ax] for ax in axes]

    fig.suptitle(title, fontsize=13)

    for r, bl in enumerate(blengths):
        for c, en in enumerate(enatives):
            ax = axes[r][c]
            key = (en, bl)
            if key in data:
                frames, values = data[key]
                ax.plot(frames, values, lw=0.9, color='steelblue')
            else:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=7, color='gray')
            if r == 0:
                ax.set_title(f'Enative={en}', fontsize=8)
            if c == 0:
                ax.set_ylabel(f'blength={bl}\n{ylabel}', fontsize=7)
            ax.tick_params(labelsize=6)

    fig.text(0.5, 0.01, 'Frame', ha='center', fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    job_to_blength = parse_master_log(MASTER_LOG)
    print(f"Loaded {len(job_to_blength)} job->blength mappings\n")

    pkl_files = glob.glob(f"{OUTPUT_DIR}/**/*.pkl", recursive=True)
    print(f"Found {len(pkl_files)} pkl files\n")

    data_max = {}  # (enative, blength) -> (frames, max_sizes)
    data_num = {}  # (enative, blength) -> (frames, num_clusters)

    for pkl_path in sorted(pkl_files):
        enative, job_id = pkl_metadata(pkl_path)
        if enative is None:
            print(f"[skip] can't parse metadata: {pkl_path}")
            continue
        blength = job_to_blength.get(job_id)
        if blength is None:
            print(f"[skip] job {job_id} not in master.log or has no blength")
            continue

        print(f"  loading  Enative={enative}  blength={blength}  job={job_id}")
        with open(pkl_path, 'rb') as f:
            d = pickle.load(f)

        frames, max_sizes, num_clusters = compute_stats(d['clusters'])
        key = (enative, blength)
        data_max[key] = (frames, max_sizes)
        data_num[key] = (frames, num_clusters)

    enatives = sorted({k[0] for k in data_max})
    blengths = sorted({k[1] for k in data_max})
    print(f"\nEnative values: {enatives}")
    print(f"blength values: {blengths}\n")

    make_panel_figure(data_max, blengths, enatives,
                      ylabel='largest cluster size',
                      title='Largest Cluster Size vs Time',
                      out_path=f"{FIGURES_DIR}/panel_max_cluster_size.png")

    make_panel_figure(data_num, blengths, enatives,
                      ylabel='# clusters (size ≥ 2)',
                      title='Number of Clusters (size ≥ 2) vs Time',
                      out_path=f"{FIGURES_DIR}/panel_num_clusters.png")

    print("Done.")
