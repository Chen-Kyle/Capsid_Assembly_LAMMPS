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
    p.add_argument('--pdb',        default=f'{HBV_ENM_PATH}/scripts/important_oligomer_pdbs/cg_ABCD_avg.pdb',
                   help='Simulation-start PDB (separated decamer)')
    p.add_argument('--traj',       default=f'{HBV_ENM_PATH}/trajectory_files/ABCD_avg_Enative/Enative_traj_files/Enative=1.5_seed=42/seg.dcd',
                   help='trajectory file for pdb')
    p.add_argument('--contactdir', default=f'{HBV_ENM_PATH}/scripts/claude_computed_contact_files',
                   help='Directory containing A_contacts.txt … D_contacts.txt')
    p.add_argument('--cutoff',    type=float, default=8.0,
                   help='Sets cutoff distance for identifying bonds')
    p.add_argument('--contacts',    type=int, default=10,
                   help='The number of contacts to be considered bonded')
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
    cols = ['resname1', 'resnum1', 'resname2', 'resnum2', 'dist', 'score', 'computed_dist']
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
    -  iface_res_data[frame_#]{B1 - C2 : ({resid1, resid2: dist_resid1_resid2}, {resid3, resid4 :...}...}
    """
    print("\nParsing contacts in frames...\n")
    iface_bonds = defaultdict(lambda: defaultdict(int))
    iface_res_data = defaultdict(lambda: defaultdict(dict))

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
                        iface_res_data[frame][iface_name][(res1, res2)] = dists[i, j]

    return iface_bonds, iface_res_data


def parse_frames_computed_cutoffs(pdb_file, traj_file, contact_dir):
    """
    Like parse_frames but uses per-residue-pair cutoff distances loaded from
    A_contacts_with_computed.txt ... D_contacts_with_computed.txt.

    The computed cutoff files are expected to have the same columns as the
    original contact files plus a final column with the per-pair cutoff distance:
        resname1 resnum1 resname2 resnum2 dist score computed_cutoff

    Returns the same two dictionaries as parse_frames:
    -  iface_bonds[frame][(segid1, segid2)]  = contact count
    -  iface_res_data[frame][iface_name][(res1, res2)] = distance
    """
    print("\nParsing pre-computed contacts in frames...\n")
    iface_bonds = defaultdict(lambda: defaultdict(int))
    iface_res_data = defaultdict(lambda: defaultdict(dict))

    stdev_bond = 2 #Gaussian stdev in the histogram is about 2-3 angstroms

    # Build per-pair cutoff lookup and pair list from the computed cutoff files
    cols = ['resname1', 'resnum1', 'resname2', 'resnum2', 'dist', 'score', 'computed_cutoff']
    computed_cutoffs = {}   # (iface, res1, res2) -> cutoff distance
    contacts = {iface: [] for iface in 'ABCD'}

    for iface in 'ABCD':
        partner = CONTACT_PARTNER_CHAIN[iface]
        clist = pd.read_csv(f"{contact_dir}/{iface}_contacts_with_computed.txt",
                            sep=r'\s+', header=None, names=cols)
        for _, row in clist.iterrows():
            res1, res2 = int(row['resnum1']), int(row['resnum2'])
            computed_cutoffs[(iface, res1, res2)] = float(row['computed_cutoff'])
            contacts[iface].append((iface, res1, partner, res2))

    u = mda.Universe(pdb_file, traj_file)
    Nframes = len(u.trajectory)
    print(f"Number of frames: {Nframes}")

    # Pre-compute all selections and attach the per-pair cutoff
    pair_selections = []
    for iface, pairs in contacts.items():
        for (chain1, res1, chain2, res2) in pairs:
            ag1 = u.select_atoms(f'resid {res1} and chainID {chain1}')
            ag2 = u.select_atoms(f'resid {res2} and chainID {chain2}')
            cutoff = computed_cutoffs[(iface, res1, res2)] + stdev_bond
            pair_selections.append((iface, res1, res2, ag1, ag2, cutoff))

    for ts in u.trajectory:
        frame = ts.frame
        for (iface, res1, res2, ag1, ag2, cutoff) in pair_selections:
            dists = mda.lib.distances.distance_array(
                ag1.positions, ag2.positions, box=u.dimensions
            )
            for i, atom1 in enumerate(ag1):
                for j, atom2 in enumerate(ag2):
                    if dists[i, j] < cutoff and atom1.segid != atom2.segid:
                        iface_bonds[frame][(atom1.segid, atom2.segid)] += 1
                        iface_name = f"{atom1.segid}-{atom2.segid}"
                        iface_res_data[frame][iface_name][(res1, res2)] = dists[i, j]

    return iface_bonds, iface_res_data


# ---------------------------------------------------------------------------
# Analysis Functions
# ---------------------------------------------------------------------------

def build_clusters(iface_bonds, contacts_per_bond):
    """
    For each frame, finds connected components of monomers linked by
    enough contacts, and records which interfaces are active in each cluster.

    Returns:
        clusters[frame] = list of dicts, each with:
            'segments'   : frozenset of segids in the cluster
            'interfaces' : dict mapping (segid1, segid2) -> contact count
                           for every active edge within the cluster
    """
    print("\nBuilding Clusters...\n")
    clusters = {}

    for frame, frame_data in iface_bonds.items():
        # Collect active edges (meet the threshold)
        active_edges = {pair: count for pair, count in frame_data.items()
                        if count >= contacts_per_bond}

        # Build adjacency from active edges
        # A dictionary containing each segid as a key and a set of its "bonded" neighbors for the values
        neighbors = defaultdict(set)
        for (s1, s2) in active_edges:
            neighbors[s1].add(s2)
            neighbors[s2].add(s1)

        # BFS to find connected components
        visited = set()
        frame_clusters = []
        for start in neighbors:
            if start in visited:
                continue
            component = set()
            queue = [start]
            while queue:
                curr = queue.pop()
                if curr in visited:
                    continue
                visited.add(curr)
                component.add(curr)
                # Set subtraction - removes all visited interfaces from the queue
                queue.extend(neighbors[curr] - visited)

            # Collect all active edges whose both endpoints are in this component
            ifaces = {pair: count for pair, count in active_edges.items()
                      if pair[0] in component and pair[1] in component}

            frame_clusters.append({
                'segments':   frozenset(component),
                'interfaces': ifaces,
            })

        clusters[frame] = frame_clusters

    return clusters


def get_cluster_data(pdb_file, traj_file, contact_dir, contacts_per_bond, output_dir):
    """
    Runs the full pipeline:
        1. parse_frames_computed_cutoffs  -> iface_bonds, iface_res_data
        2. build_clusters                 -> clusters

    Saves all three to a pickle file at output_dir/cluster_data.pkl as:
        {
            'iface_bonds':    iface_bonds,
            'iface_res_data': iface_res_data,
            'clusters':       clusters,
        }

    Returns the same three objects.
    """
    import pickle

    iface_bonds, iface_res_data = parse_frames_computed_cutoffs(pdb_file, traj_file, contact_dir)
    clusters = build_clusters(iface_bonds, contacts_per_bond)

    output_path = f"{output_dir}/cluster_data.pkl"
    with open(output_path, "wb") as f:
        pickle.dump({
            'iface_bonds':    iface_bonds,
            'iface_res_data': iface_res_data,
            'clusters':       clusters,
        }, f)
    print(f"Saved cluster data to {output_path}")

    return iface_bonds, iface_res_data, clusters


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
    iface_bonds, iface_res_data, clusters = get_cluster_data(
        args.pdb, args.traj, args.contactdir, args.contacts, args.output_dir
    )
    #plot_iface_contacts(iface_bonds, args.output_dir)
    #plot_dists(iface_res_data, "B1-C1", args.output_dir)
