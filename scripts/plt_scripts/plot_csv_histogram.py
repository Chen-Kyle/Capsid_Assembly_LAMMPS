"""
Plots the spike angle distribution grouped by dimer site (A/B/C/D site),
same as get_spike_angle.py's plot_spike_angle_data -- but reads directly
from a complete_cluster_data.pkl produced by full_traj_analysis.py, which
already has the spike angle for every active interface at every frame in
pkl_data['interface_data'], instead of recomputing it from the trajectory.
"""

import os
import re
import argparse
import pickle
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

HBV_ENM_PATH = os.environ.get("HBV_ENM_PATH", "/home/kyle/2026_Research/HBV_enm")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default=f"{HBV_ENM_PATH}/scripts/lammps_out/binding_angles.csv",
                    help='Path to the csv file to plot')
    p.add_argument('--output_dir', default=f'{HBV_ENM_PATH}/raw_data',
                    help='Directory to save the output figure to')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def load_csv(csv_file):
    """
    Loads the csv_file with pandas
    """
    return pd.read_csv(csv_file)

def format_chain_name(segids):
    """
    Formats the segids so that they will have a dash in between the chain letters
    """
    chains = ''.join(letter for letter in segids if letter.isalpha())
    if chains == "DC":
        chains = "CD"
    if chains == "BA":
        chains = "AB"
    chain = chains[0] + '-' + chains[1]
    return chain

def normalize_string(s):
    return s.lower().replace(' ', '_')

def get_enative(path):
    """
    Extracts the Enative float from a file path containing 'Enative=<value>'.
    """
    match = re.search(r'Enative=([0-9]+(?:\.[0-9]+)?)', path)
    return float(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------

def build_histogram_data(df):
    """
    Reads the csv file and then creates a dictionary which is keyed
    by the site type/dimer and then then contains an array of data values
    for that site type/dimer
    """
    histogram_data = defaultdict(list)

    for index, row in df.iterrows():
        label = format_chain_name(row.iloc[1])
        value_for_plotting = row.iloc[2]
        histogram_data[label].append(value_for_plotting)

    print(f"Histogram data: {histogram_data}")

    return histogram_data

def plot_histogram_data(plotting_data, csv_data_frame, output_dir):
    """
    Plots the data and automatically renames the title and axes based
    off of the header

    Saves the plot to the output_dir with a name also from the header
    """

    fig, ax = plt.subplots()
    for label, values in plotting_data.items():

        x_min = np.min(values)
        x_max = np.max(values)
        print(f"xmin: {x_min}, xmax: {x_max} for label: {label}")

        mean = np.mean(values)
        stdev = np.std(values)
        weights = np.ones_like(values) / len(values)
        line = ax.hist(values, bins=60, alpha=0.6, weights=weights,
                       label=f'{label} (mean={mean:.3f}, std={stdev:.3f})')
        ax.axvline(mean, color=line[2][0].get_facecolor(), linestyle='--', linewidth=1)

    headers = csv_data_frame.columns.tolist()
    ax.set_xlabel(f"{headers[2]} (rads)", fontsize=14)
    ax.set_ylabel('Relative Frequency', fontsize=14)
    # enative_str = f' — Enative={enative}' if enative is not None else ''
    ax.set_title(f'{headers[2]} Distribution by {headers[1]}', fontsize=18)
    ax.legend(fontsize=12)
    fig.tight_layout()

    out_label = normalize_string(headers[2])
    out_path = f"{output_dir}/{out_label}_histogram.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args()
    csv_data = pd.read_csv(args.csv)
    plotting_data = build_histogram_data(csv_data)
    plot_histogram_data(plotting_data, csv_data, args.output_dir)
