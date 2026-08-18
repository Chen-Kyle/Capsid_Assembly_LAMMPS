"""
Plots cluster size vs time using the data from a get_clusters.py csv output
Good for using on the outputs of:
 - get_clusters.py (all_cluster.csv, well_formed_clusters.csv)

Input arguements:
    path_to_csv_file               -- csv containing Frame#, ClusterID, Cluster Size
    type                           -- "all" or "wellformed", picks the default csv
                                       and labels the output figures

Output
    Two figures, saved to output_dir:
     - largest cluster size vs time
     - cluster size vs time, one line per persistent ClusterID
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

HBV_ENM_PATH = os.environ.get("HBV_ENM_PATH", "/home/kyle/2026_Research/HBV_enm")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default=None,
                    help='Path to the csv file to plot. Defaults to '
                         'all_cluster.csv or well_formed_clusters.csv under '
                         'scripts/lammps_out, based on --type')
    p.add_argument('--type', choices=['all', 'wellformed'], default='all',
                    help='Which cluster csv to plot: "all" or "wellformed", default is all')
    p.add_argument('--output_dir', default=f'{HBV_ENM_PATH}/raw_data',
                    help='Directory to save the output figures to')
    args = p.parse_args()

    if args.csv is None:
        default_name = 'well_formed_clusters.csv' if args.type == 'wellformed' else 'all_cluster.csv'
        args.csv = f'{HBV_ENM_PATH}/scripts/lammps_out/{default_name}'

    return args


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def load_csv(csv_file):
    """
    Loads the csv_file with pandas
    """
    return pd.read_csv(csv_file)


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------

def build_max_size_over_time(df):
    """
    Groups rows by Frame# and takes the largest cluster size present in that
    frame.

    Returns two parallel arrays, sorted by frame: frames, max_sizes
    """
    frame_col, size_col = df.columns[0], df.columns[2]
    grouped = df.groupby(frame_col)[size_col].max().sort_index()
    return grouped.index.to_numpy(), grouped.to_numpy()


def build_size_by_cluster_id(df):
    """
    Groups rows by ClusterID for plotting one line per persistent cluster.

    Returns {cluster_id: (frames, sizes)}, each sorted by frame.
    """
    frame_col, id_col, size_col = df.columns[0], df.columns[1], df.columns[2]
    data = {}
    for cluster_id, group in df.groupby(id_col):
        group = group.sort_values(frame_col)
        data[cluster_id] = (group[frame_col].to_numpy(), group[size_col].to_numpy())
    return data


def plot_max_cluster_size(df, output_dir, csv_path, type_label):
    """
    Plots the largest cluster size in each frame vs time.
    """
    frames, max_sizes = build_max_size_over_time(df)
    size_col = df.columns[2]

    fig, ax = plt.subplots()
    ax.plot(frames, max_sizes, marker='o', markersize=2, linewidth=1)

    ax.set_xlabel('Frame#', fontsize=14)
    ax.set_ylabel(size_col, fontsize=14)
    ax.set_title(f'Largest {size_col} ({type_label}) vs Time', fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.text(0.5, 0.02, f"Source: {csv_path}", ha='center', fontsize=8, color='gray')

    out_path = f"{output_dir}/{type_label}_max_cluster_size_vs_time.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.show()


def plot_cluster_size_by_id(df, output_dir, csv_path, type_label):
    """
    Plots cluster size vs time, with each ClusterID drawn as its own line/color.
    """
    size_by_id = build_size_by_cluster_id(df)
    size_col = df.columns[2]

    fig, ax = plt.subplots()
    for cluster_id, (frames, sizes) in sorted(size_by_id.items()):
        ax.plot(frames, sizes, label=f'Cluster {cluster_id}', linewidth=1)

    ax.set_xlabel('Frame#', fontsize=14)
    ax.set_ylabel(size_col, fontsize=14)
    ax.set_title(f'{size_col} ({type_label}) vs Time by ClusterID', fontsize=16)
    ax.legend(fontsize=8, ncol=2)
    ax.tick_params(axis='both', labelsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.text(0.5, 0.02, f"Source: {csv_path}", ha='center', fontsize=8, color='gray')

    out_path = f"{output_dir}/{type_label}_cluster_size_by_id_vs_time.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args()
    csv_data = load_csv(args.csv)

    plot_max_cluster_size(csv_data, args.output_dir, args.csv, args.type)
    plot_cluster_size_by_id(csv_data, args.output_dir, args.csv, args.type)
