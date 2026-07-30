"""
get_spike_angle.py
Must be used only on specific pdb files:
- abcd_capsid.pdb
...

Input arguements:
    path_to_pdb_file                    -- atom positions + bond topology
    path_to_trajectory_file             -- trajectory data
    output_directory_path               -- output directory

Output
    An HDF5 file containing the distance between every residue
    to another residue to the other dimer for every frame of the simulation,
    plus the mean distance matrix across all frames
"""

import argparse
import os
import MDAnalysis as mda
import numpy as np
import h5py
import pickle
import matplotlib.pyplot as plt
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

CONTACTS_PER_BOND = 10
HBV_ENM_PATH = os.environ.get("HBV_ENM_PATH", "/home/kyle/2026_Research/HBV_enm")

# Maps the monomer from one dimer to its other monomer pair
PARTNER_MAP = {'A' : 'B', 'B' : 'A', 'C' : 'D', 'D' : 'C'}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pkl',
                   help='Path to the pickle file', default=f'{HBV_ENM_PATH}/scripts/binding_angles_lammps_out/seg_cluster_data.pkl')
    p.add_argument('--out',
                   help='Directory path to send the data to', default=f'{HBV_ENM_PATH}/raw_data')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def load_pickle(path):
    """
    Loads the data from the pickle file
    """
    with open(path, 'rb') as f:
        return pickle.load(f)

def get_partner(residue_name):
    """
    Returns the monomer that is on the other side of the dimer
    """
    res_partner_chain = PARTNER_MAP[residue_name[0]]
    res_partner = res_partner_chain + residue_name[1:]
    return res_partner

def get_segids(interface_name):
    """
    Extracts the chainIDs out of the interface name string
    """
    segids = interface_name.split('-')
    return segids

def get_dimer_group(dimer_name):
    """
    Gets just the chain letters from a dimer name (e.g. 'A1-B1') and orders
    the result to be either 'AB' or 'CD', so dimers can be grouped regardless
    of monomer number or letter order.
    """
    chains = ''.join(letter for letter in dimer_name if letter.isalpha())
    if chains == "DC":
        chains = "CD"
    if chains == "BA":
        chains = "AB"
    return chains

def dot_product(vector1, vector2):
    """
    Takes the dot product of two 3D vectors: vector1 * vector2
    """
    dot_product = vector1[0]*vector2[0] +  vector1[1]*vector2[1] + vector1[2]*vector2[2]
    return dot_product

def normalize_vector(vector):
    """
    Normalizes a 3D vector
    """
    magnitude = (vector[0]**2 + vector[1]**2 + vector[2]**2)**(1/2)
    normalized_vector = vector/magnitude
    return normalized_vector

def detect_dimer_list(u_sim):
    """
    Auto-detect which AB and CD dimers are present in the PDB from its segIDs.
    Returns a list like ['A1B1', 'C1D1', 'A2B2', ...] ordered by monomer number,
    interleaved AB/CD as in the original decamer setup.
    """
    segids = set(u_sim.select_atoms('name CA').segids)
    segidNums = set(int(s[1:]) for s in segids)
    n_dimers = max(segidNums) + 1

    print(f"segids:{segids}")
    print(f"Numbers of dimers:{n_dimers}")

    dimer_list = []
    for i in range(1, n_dimers):
        if f'A{i}' in segids and f'B{i}' in segids:
            dimer_list.append(f'A{i}-B{i}')
        if f'C{i}' in segids and f'D{i}' in segids:
            dimer_list.append(f'C{i}-D{i}')
    return dimer_list



# ---------------------------------------------------------------------------
# Spike Angle Calculation
# ---------------------------------------------------------------------------

def calculate_spike_angle(u, interface_name, frameNum):
    """
    Calculates the spike angle of a specific interface given its name and frame #.
    Returns the spike angle in radians.
    """
    u.trajectory[frameNum]

    distance_vectors = []
    segids = get_segids(interface_name)
    for segid in segids:
        atom_73 = u.select_atoms(f'resid 73 and segid {segid}')
        atom_58 = u.select_atoms(f'resid 58 and segid {segid}')

        pos1 = atom_73.positions[0]
        pos2 = atom_58.positions[0]
        distance_vector = normalize_vector(pos1 - pos2)
        distance_vectors.append(distance_vector)

    dotted_value = dot_product(distance_vectors[0], distance_vectors[1])
    spike_angle = np.arccos(np.clip(dotted_value, -1.0, 1.0))
    return spike_angle

def calculate_all_spike_angles(u, pkl_data):
    """
    For every frame, iterates over all active inter-dimer interfaces and
    computes the spike angle at each one.

    The spike angle at interface "X_i - Y_j" is the angle between:
        v1 = spike-tip direction of monomer X_i (resid 73 -> resid 58)
        v2 = spike-tip direction of monomer Y_j (resid 73 -> resid 58)

    Returns:
        spike_angle_data[interface_name][frame] = angle (radians)
    """
    from collections import defaultdict

    iface_bonds = pkl_data['iface_bonds']

    # Pre-build atom selections for all segids once
    segids = set(u.select_atoms('name CA').segids)
    atom_sel_73 = {seg: u.select_atoms(f'resid 73 and segid {seg}') for seg in segids}
    atom_sel_58 = {seg: u.select_atoms(f'resid 58 and segid {seg}') for seg in segids}

    spike_angle_data = defaultdict(dict)

    for ts in u.trajectory:
        frame = ts.frame
        frame_data = iface_bonds.get(frame, {})

        for (seg1, seg2), count in frame_data.items():
            if count < CONTACTS_PER_BOND:
                continue
            if any(s not in atom_sel_73 or s not in atom_sel_58 for s in (seg1, seg2)):
                continue

            v1 = normalize_vector(atom_sel_73[seg1].positions[0] - atom_sel_58[seg1].positions[0])
            v2 = normalize_vector(atom_sel_73[seg2].positions[0] - atom_sel_58[seg2].positions[0])

            angle = np.arccos(np.clip(dot_product(v1, v2), -1.0, 1.0))
            spike_angle_data[f"{seg1}-{seg2}"][frame] = angle

    return spike_angle_data

def plot_spike_angle_data(spike_angle_data):
    """
    Plots spike angle distributions for all 4 sites (A, B, C, D) as overlapping
    histograms. Each site pools all angles from all interfaces of that type across
    all frames.
    """
    site_label = {
        'AA':         'A site',
        'BC': 'B site', 'CB': 'B site',
        'CD': 'C site', 'DC': 'C site',
        'DB': 'D site', 'BD': 'D site',
    }

    site_angles = {'A site': [], 'B site': [], 'C site': [], 'D site': []}

    for interface, frame_angles in spike_angle_data.items():
        group = get_dimer_group(interface)
        site = site_label.get(group)
        if site is None:
            continue
        site_angles[site].extend(frame_angles.values())

    all_arr = np.array([v for angles in site_angles.values() for v in angles])
    pad = 0.15
    x_min = np.percentile(all_arr, 0.5) - pad
    x_max = np.percentile(all_arr, 99.5) + pad

    fig, ax = plt.subplots()
    for site, angles in site_angles.items():
        if not angles:
            print(f"No data for {site}")
            continue
        values = np.array(angles)
        mean = np.mean(values)
        stdev = np.std(values)
        weights = np.ones_like(values) / len(values)
        line = ax.hist(values, bins=60, alpha=0.6, weights=weights,
                       range=(x_min, x_max),
                       label=f'{site} (mean={mean:.3f}, std={stdev:.3f})')
        ax.axvline(mean, color=line[2][0].get_facecolor(), linestyle='--', linewidth=1)

    print(f"x_min: {x_min} and x_max: {x_max}")
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel('Spike angle (rad)')
    ax.set_ylabel('Relative Frequency')
    ax.legend()
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    args = parse_args()
    pkl_data = load_pickle(args.pkl)
    pdb = pkl_data['pdb_file']
    traj = pkl_data['traj_file']
    try:
        u = mda.Universe(pdb, traj)
    except:
        basename = os.path.basename(args.pkl)
        traj = args.pkl.replace(basename, "seg.dcd")
        u = mda.Universe(pdb, traj)

    spike_angle_data = calculate_all_spike_angles(u, pkl_data)
    plot_spike_angle_data(spike_angle_data)
