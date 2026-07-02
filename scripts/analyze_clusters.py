"""
analyze_clusters.py
Contains the functions for running analysis on cluster data pickle files

Input arguements:
    path_to_pickle_file                 -- pickle file to run analysis on
    output_directory_path               -- output directory

Output
    There are two main functions: parse_computed_cutoff and build_clusters

    parse_computed_cutoffs returns: 2 dictionaries containing native contact bond data:
    -  iface_contacts[frame_#][B1, C2] = (number of bonds in the B1-C2 interface in frame_#)
    -  type_data[frame_#]{B1 - C2 : ([resid1, resid2], [resid3, resid4]...}

    build_clusters returns: A dictionary containing cluster info per frame
    -   clusters[frame_#][list indexed by 0,1,2...]{segments: (all seg_ids
        in cluster), interfaces: (all interfaces in cluster)}
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
import MDAnalysis as mda
import matplotlib.pyplot as plt
from collections import defaultdict

HBV_ENM_PATH = os.environ.get("HBV_ENM_PATH", "/home/kyle/2026_Research/HBV_enm")

# ---------------------------------------------------------------------------
# Supplementary Functions + CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--file',
                   help='File path of the cluster pickle file to be analyzed')
    p.add_argument('--output_dir',     default=f'{HBV_ENM_PATH}/raw_data/cluster_data',
                   help='Output directory for bond analysis data')
    p.add_argument("--interactive",        action="store_true",
                   help="Turns on interactive mode for inputting which what analysis function to use")
    return p.parse_args()


def load_pickle(path):
    """
    Loads the data from the pickle file
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_probability(iface_contacts, contacts_per_bond):
    """
    Returns the fraction of time each interface was bonded given
    the number of contacts_per_bond threshold.

    Fraction_bonded = (frames_bonded) / (total_frames)
    """
    print("\nRunning get_probability")
    total_frames = max(iface_contacts.keys()) + 1

    # Count frames where each interface had >= contacts_per_bond contacts
    frames_bonded = defaultdict(int)
    for frame_data in iface_contacts.values():
        for interface, count in frame_data.items():
            if count >= contacts_per_bond:
                frames_bonded[interface] += 1

    probability = {iface: frames_bonded[iface] / total_frames
                   for iface in frames_bonded}
    print(f"    probability: {probability}")
    return probability


# ---------------------------------------------------------------------------
# Plotting Functions
# ---------------------------------------------------------------------------

def plot_iface_contacts(iface_contacts, output_dir):
    """
    Plot each unique interface as a line with smart formatting
    """
    all_frames = list(range(max(iface_contacts.keys()) + 1))

    # Collect all unique interface names
    all_ifaces = set()
    for frame_data in iface_contacts.values():
        all_ifaces.update(frame_data.keys())

    step = 10
    frames = all_frames[::step]
    n_ifaces = len(all_ifaces)
    
    print(f"Plotting {n_ifaces} interfaces across {len(frames)} frames")
    
    # IMPROVEMENT 1: Adaptive figure size based on data
    # Wider if many interfaces, narrower if few
    fig_width = max(16, 20 + (n_ifaces / 50))  # Scales with complexity
    
    fig, ax = plt.subplots(figsize=(fig_width, 7), dpi=100)  # Lower DPI, larger fig
    
    # IMPROVEMENT 2: Better coloring scheme
    colors = plt.cm.tab20c(np.linspace(0, 1, min(n_ifaces, 20)))
    if n_ifaces > 20:
        colors = plt.cm.hsv(np.linspace(0, 0.9, n_ifaces))
    
    # IMPROVEMENT 3: Plot with better styling
    for idx, iface in enumerate(all_ifaces):
        bonds = pd.Series([iface_contacts.get(f, {}).get(iface, 0) for f in frames])
        smoothed = bonds.rolling(window=50, center=True, min_periods=1).mean()
        color = colors[idx % len(colors)]
        ax.plot(frames, bonds, linewidth=0.5, alpha=0.2, color=color)
        ax.plot(frames, smoothed,
                label=iface,
                linewidth=1.5,
                alpha=0.8,
                color=color)
    
    # IMPROVEMENT 4: Better axis labels and title
    ax.set_xlabel("Frame", fontsize=12, fontweight='bold')
    ax.set_ylabel("Number of Contacts", fontsize=12, fontweight='bold')
    ax.set_title(f"Native Contacts per Interface ({n_ifaces} unique)", 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')  # Subtle grid for readability
    
    # IMPROVEMENT 5: Smart legend handling
    if n_ifaces <= 20:
        # Small number of interfaces: show legend
        ax.legend(loc='upper left', fontsize=8)
    else:
        # Too many interfaces: sample legend or omit
        # Option A: Show every Nth interface
        sample_indices = np.linspace(0, n_ifaces - 1, min(15, n_ifaces), dtype=int)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([handles[i] for i in sample_indices],
                  [labels[i] for i in sample_indices],
                  bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/iface_contacts.png", dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/iface_contacts.png")
    plt.show()
    return


def plot_dists(iface_res_data, interface_name, output_dir):
    """
    4x5 panel plot of distance distributions for each native contact
    in the given interface across all frames.

    iface_res_data : returned from parse_frames
    interface_name : e.g. "B1-C1"
    """
    all_ifaces = set()
    for frame_data in iface_res_data.values():
        all_ifaces.update(frame_data.keys())
    print("  Available interfaces:", sorted(all_ifaces))
    if interface_name is None:
        interface_name = input("  Interface name: ")
    print(f'Plotting native contact distances for: {interface_name}')
    contact_dists = defaultdict(list)
    for frame_data in iface_res_data.values():
        if interface_name in frame_data:
            for (res1, res2), dist in frame_data[interface_name].items():
                contact_dists[(res1, res2)].append(dist)

    if not contact_dists:
        print(f"No data found for interface '{interface_name}'")
        return

    contacts = sorted(contact_dists.keys())
    if len(contacts) > 20:
        print(f"Warning: {len(contacts)} contacts found, only plotting first 20")

    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()

    for idx, (res1, res2) in enumerate(contacts[:20]):
        ax = axes[idx]
        dists = contact_dists[(res1, res2)]
        median = np.median(dists)
        ax.hist(dists, bins=30, color='steelblue', edgecolor='white', linewidth=0.5)
        ax.axvline(median, color='red', linewidth=1.2, linestyle='--', label=f'median={median:.1f}Å')
        ax.legend(fontsize=6, loc='upper right')
        ax.set_title(f"Res {res1} - Res {res2}", fontsize=9, fontweight='bold', pad=4)
        ax.set_xlabel("Distance (Å)", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.tick_params(labelsize=7)

    for idx in range(len(contacts), 20):
        axes[idx].set_visible(False)

    fig.suptitle(f"Native Contact Distance Distributions: {interface_name}",
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.subplots_adjust(hspace=0.55)
    plt.savefig(f"{output_dir}/dists_{interface_name.replace('-', '_')}.png", dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/dists_{interface_name.replace('-', '_')}.png")
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    data = load_pickle(args.file)

    if args.interactive:
        functions = {
            1: ("get_probability",     lambda: get_probability(data['iface_bonds'], contacts_per_bond=10)),
            2: ("plot_iface_contacts", lambda: plot_iface_contacts(data['iface_bonds'], args.output_dir)),
            3: ("plot_dists",          lambda: plot_dists(data['iface_res_data'], None, args.output_dir)),
        }
        for n, (name, fn) in functions.items():
            run = input(f"[{n}] {name} — run? (y/n): ")
            if run.strip().lower() == "y":
                fn()
