"""
detect_bonds.py
Contains the functions for detecting bonds via native contacts

Input arguements:
    cutoff (in Angstroms)               -- default = 7Å
    path_to_pdb_file                    -- atom positions + bond topology
    path_to_trajectory_file             -- trajectory data
    {A,B,C,D}_contacts.txt folder       -- native contacts data

Output
    Returns 2 dictionaries containing native contact bond data:
    -  iface_contacts[frame_#][B1, C2] = (number of bonds in the B1-C2 interface in frame_#)
    -  type_data[frame_#]{B1 - C2 : ([resid1, resid2], [resid3, resid4]...}
"""

import argparse
import os
import numpy as np
import pandas as pd
import MDAnalysis as mda
import matplotlib.pyplot as plt
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")

HBV_ENM_PATH = os.environ.get("HBV_ENM_PATH", "/home/kyle/2026_Research/HBV_enm")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pdb',        default=f'{HBV_ENM_PATH}/scripts/important_oligomer_pdbs/cg_ABCD_separate.pdb',
                   help='Simulation-start PDB (separated decamer)')
    p.add_argument('--traj',       default=f'{HBV_ENM_PATH}/trajectory_files/ABCD_seg.dcd',
                   help='trajectory file for pdb')
    p.add_argument('--contactdir', default=f'{HBV_ENM_PATH}/scripts/contact_files',
                   help='Directory containing A_contacts.txt … D_contacts.txt')
    p.add_argument('--cutoff',    type=float, default=8.0,
                   help='Sets cutoff distance for identifying bonds')
    p.add_argument('--output_dir',     default=f'{HBV_ENM_PATH}/raw_data',
                   help='Output directory for bond analysis data')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Bond Constuction Functions
# ---------------------------------------------------------------------------

# Right-column chain type for each contacts file (derived from spatial query exclusions)
#   A_contacts.txt : A chain contacts A chain
#   B_contacts.txt : B chain contacts C chain  (D excluded in original query)
#   C_contacts.txt : C chain contacts D chain  (A and B excluded, not-D{mn} leaves D other)
#   D_contacts.txt : D chain contacts B chain  (C and A excluded)
CONTACT_PARTNER_CHAIN = {'A': 'A', 'B': 'C', 'C': 'D', 'D': 'B'}

def build_native_contacts(contact_dir):
    """
    Returns contacts dict keyed by interface ('A','B','C','D').
    Each value is a list of (chain1, resnum1, chain2, resnum2) tuples.

    Any atom on chain1 at resnum1 can attract any atom on chain2 at resnum2
    regardless of which specific monomer it belongs to. This allows 
    self-assembly with arbitrary numbers of dimers.
    """
    cols = ['resname1', 'resnum1', 'resname2', 'resnum2', 'dist', 'score']
    contacts = {iface: [] for iface in 'ABCD'}
    for iface in 'ABCD':
        partner = CONTACT_PARTNER_CHAIN[iface]
        clist = pd.read_csv(f"{contact_dir}/{iface}_contacts.txt",
                            sep=r'\s+', header=None, names=cols)
        for _, row in clist.iterrows():
            contacts[iface].append(
                (iface, int(row['resnum1']), partner, int(row['resnum2']))
            )
    return contacts

def parse_frames(cutoff, pdb_file, traj_file, contact_dir):
    """
    Loops through the trajectory file to look for native contacts

    Returns 2 dictionaries containing native contact bond data:
    -  iface_type_bonds[frame_#][B1, C2] = (number of bonds in the B1-C2 interface in frame_#)
    -  iface_res_data[frame_#]{B1 - C2 : ([resid1, resid2], [resid3, resid4]...}
    """
    iface_bonds = defaultdict(lambda: defaultdict(int))
    iface_res_data = defaultdict(lambda: defaultdict(list))

    u = mda.Universe(pdb_file, traj_file)
    
    contacts = build_native_contacts(contact_dir)
    Nframes = len(u.trajectory)
    print(f"Number of frames: {Nframes}")

    # Pre-compute all selections once
    pair_selections = []
    for iface, pairs in contacts.items():
        for (chain1, res1, chain2, res2) in pairs:
            ag1 = u.select_atoms(f'resid {res1} and chainID {chain1}')
            ag2 = u.select_atoms(f'resid {res2} and chainID {chain2}')
            pair_selections.append((iface, res1, res2, ag1, ag2))

    for ts in u.trajectory:
        frame = ts.frame
        for (iface, res1, res2, ag1, ag2) in pair_selections:
            dists = mda.lib.distances.distance_array(
                ag1.positions, ag2.positions, box=u.dimensions
            )
            for i, atom1 in enumerate(ag1):
                for j, atom2 in enumerate(ag2):
                    if dists[i, j] < cutoff and atom1.segid != atom2.segid:
                        iface_bonds[frame][(atom1.segid, atom2.segid)] += 1
                        iface_name = f"{atom1.segid}-{atom2.segid}"
                        iface_res_data[frame][iface_name].append((res1, res2))

    return iface_bonds, iface_res_data


# ---------------------------------------------------------------------------
# Analysis Functions
# ---------------------------------------------------------------------------

def get_probability(iface_contacts, contacts_per_bond):
    """
    Returns the fraction of time each interface was bonded given
    the number of contacts_per_bond threshold.

    Fraction_bonded = (frames_bonded) / (total_frames)
    """
    total_frames = max(iface_contacts.keys()) + 1

    # Count frames where each interface had >= contacts_per_bond contacts
    frames_bonded = defaultdict(int)
    for frame_data in iface_contacts.values():
        for interface, count in frame_data.items():
            if count >= contacts_per_bond:
                frames_bonded[interface] += 1

    probability = {iface: frames_bonded[iface] / total_frames
                   for iface in frames_bonded}
    return probability

def plot_iface_contacts(iface_contacts):
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
    
    # For debugging
    n_contacts = iface_contacts[0][("B1", "C1")]
    print(f"Number of contacts:{n_contacts}")

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
        bonds = pd.Series([iface_contacts[f].get(iface, 0) for f in frames])
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
    plt.savefig("iface_contacts.png", dpi=150, bbox_inches='tight')
    print(f"Saved to iface_contacts.png")
    plt.show()
    return


if __name__ == "__main__":
    
    args = parse_args()
    iface_bonds, iface_res_data = parse_frames(args.cutoff, args.pdb, args.traj, args.contactdir)

    # print(iface_contacts)
    # print(type_data)
    plot_iface_contacts(iface_bonds)
    # print(iface_bonds)
