"""
analyze_last_frame_clusters.py

Standalone, single-frame variant of full_traj_analysis.py's pipeline.
Instead of walking every frame of a trajectory (slow for long runs), this
loads and analyzes ONLY the LAST frame of a given seg.dcd: native contacts,
binding angle, spike angle, and well-formed clusters, with configurable
angle thresholds.

This script is self-contained (no dependency on full_traj_analysis.py) so
its logic/thresholds can be iterated on independently.

Usage:
    python analyze_last_frame_clusters.py --pdb <pdb> --traj <seg.dcd> \
        --contacts 20 --binding_range 0.0,1.5 --spike_range 0.0,1.0
"""

import os
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance

import warnings
warnings.filterwarnings("ignore")

HBV_ENM_PATH = os.environ.get("HBV_ENM_PATH", "/home/kyle/2026_Research/HBV_enm")

# Right-column chain type for each contacts file (same convention as full_traj_analysis.py)
CONTACT_PARTNER_CHAIN = {'A': 'A', 'B': 'C', 'C': 'D', 'D': 'B'}

# Maps the monomer from one dimer to its other monomer pair
DIMER_MAP = {'A': 'B', 'B': 'A', 'C': 'D', 'D': 'C'}


def get_partner(residue_name):
    """
    Returns the monomer that is on the other side of the dimer.
    """
    return DIMER_MAP[residue_name[0]] + residue_name[1:]


def normalize_vector(vector):
    return vector / np.linalg.norm(vector)


def dot_product(v1, v2):
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]


def minimum_image(diff, box):
    """
    Applies the minimum-image convention to a displacement vector. Without
    this, a displacement between two atoms that got wrapped into different
    periodic images (e.g. two chains split across a box boundary) can come
    out with close to the wrong sign/direction.
    """
    return diff - box * np.round(diff / box)


# ---------------------------------------------------------------------------
# Native contact loading
# ---------------------------------------------------------------------------

def load_native_contacts(contact_dir):
    """
    Returns (contacts, computed_cutoffs):
        contacts[iface]                        = list of (chain1, res1, chain2, res2)
        computed_cutoffs[(iface, res1, res2)]  = per-pair cutoff distance (A)
    """
    cols = ['resname1', 'resnum1', 'resname2', 'resnum2', 'dist', 'score', 'computed_cutoff']
    contacts = {iface: [] for iface in 'ABCD'}
    computed_cutoffs = {}

    for iface in 'ABCD':
        partner = CONTACT_PARTNER_CHAIN[iface]
        clist = pd.read_csv(f"{contact_dir}/{iface}_contacts_with_computed.txt",
                            sep=r'\s+', header=None, names=cols)
        for _, row in clist.iterrows():
            res1, res2 = int(row['resnum1']), int(row['resnum2'])
            computed_cutoffs[(iface, res1, res2)] = float(row['computed_cutoff'])
            contacts[iface].append((iface, res1, partner, res2))

    return contacts, computed_cutoffs


# ---------------------------------------------------------------------------
# Single-frame contact + angle computation
# ---------------------------------------------------------------------------

def last_frame_contacts(u, contacts, computed_cutoffs, stdev_bond=3):
    """
    Computes native-contact counts between segid pairs at the CURRENT frame
    of u (caller must seek to the desired frame first).

    Returns iface_bonds: dict (segid1, segid2) -> contact count
    """
    box = u.dimensions[:3]
    iface_bonds = defaultdict(int)

    for iface, pairs in contacts.items():
        for (chain1, res1, chain2, res2) in pairs:
            ag1 = u.select_atoms(f'resid {res1} and chainID {chain1}')
            ag2 = u.select_atoms(f'resid {res2} and chainID {chain2}')
            if len(ag1) == 0 or len(ag2) == 0:
                continue
            cutoff = computed_cutoffs[(iface, res1, res2)] + 2 * stdev_bond

            pair_idx = capped_distance(ag1.positions, ag2.positions,
                                       max_cutoff=cutoff, box=u.dimensions,
                                       return_distances=False)
            for (i, j) in pair_idx:
                atom1, atom2 = ag1[i], ag2[j]
                if atom1.segid == atom2.segid:
                    continue
                diff = minimum_image(ag1.positions[i] - ag2.positions[j], box)
                dist = np.linalg.norm(diff)
                if dist < cutoff:
                    iface_bonds[(atom1.segid, atom2.segid)] += 1

    return dict(iface_bonds)


def last_frame_angles(u, iface_bonds, contacts_per_bond):
    """
    Computes binding_angle and spike_angle for every (segid1, segid2) edge in
    iface_bonds meeting contacts_per_bond, at the CURRENT frame of u.

    Returns interface_data: dict "segid1-segid2" -> {'binding_angle', 'spike_angle'}
    """
    box = u.dimensions[:3]
    segids = set(u.select_atoms('name CA').segids)
    sel_132 = {seg: u.select_atoms(f'resid 132 and segid {seg}') for seg in segids}
    sel_73  = {seg: u.select_atoms(f'resid 73 and segid {seg}') for seg in segids}
    sel_58  = {seg: u.select_atoms(f'resid 58 and segid {seg}') for seg in segids}

    interface_data = {}
    for (segid1, segid2), count in iface_bonds.items():
        if count < contacts_per_bond:
            continue
        partner1, partner2 = get_partner(segid1), get_partner(segid2)
        if any(s not in sel_132 for s in (segid1, segid2, partner1, partner2)):
            continue

        diff1_bind = minimum_image(sel_132[partner1].positions[0] - sel_132[segid1].positions[0], box)
        diff2_bind = minimum_image(sel_132[partner2].positions[0] - sel_132[segid2].positions[0], box)
        binding_angle = np.arccos(np.clip(
            dot_product(normalize_vector(diff1_bind), normalize_vector(diff2_bind)), -1.0, 1.0))

        diff1_spike = minimum_image(sel_73[segid1].positions[0] - sel_58[segid1].positions[0], box)
        diff2_spike = minimum_image(sel_73[segid2].positions[0] - sel_58[segid2].positions[0], box)
        spike_angle = np.arccos(np.clip(
            dot_product(normalize_vector(diff1_spike), normalize_vector(diff2_spike)), -1.0, 1.0))

        interface_data[f"{segid1}-{segid2}"] = {
            'binding_angle': binding_angle,
            'spike_angle':   spike_angle,
        }

    return interface_data


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def build_well_formed_clusters(iface_bonds, interface_data, contacts_per_bond,
                                binding_angle_range, spike_angle_range):
    """
    Connected-components clustering restricted to edges that meet the contact
    threshold AND have binding_angle/spike_angle within the given ranges.

    Returns a list of dicts: [{'segments': frozenset(...), 'interfaces': {...}}, ...]
    """
    active_edges = {}
    for (s1, s2), count in iface_bonds.items():
        if count < contacts_per_bond:
            continue
        angles = interface_data.get(f"{s1}-{s2}")
        if angles is None:
            continue
        if not (binding_angle_range[0] <= angles['binding_angle'] <= binding_angle_range[1]):
            continue
        if not (spike_angle_range[0] <= angles['spike_angle'] <= spike_angle_range[1]):
            continue
        active_edges[(s1, s2)] = count

    neighbors = defaultdict(set)
    for (s1, s2) in active_edges:
        neighbors[s1].add(s2)
        neighbors[s2].add(s1)
        neighbors[s1].add(get_partner(s1))
        neighbors[s2].add(get_partner(s2))

    visited = set()
    clusters = []
    for start in list(neighbors):
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
            queue.extend(neighbors.get(curr, set()) - visited)

        ifaces = {pair: count for pair, count in active_edges.items()
                  if (pair[0] in component and pair[1] in component)
                  and not (pair[1] == get_partner(pair[0]))}
        clusters.append({'segments': frozenset(component), 'interfaces': ifaces})

    return clusters


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def analyze_last_frame(pdb_file, traj_file, contact_dir, contacts_per_bond,
                        binding_angle_range=(0.0, 1.5), spike_angle_range=(0.0, 1.5)):
    """
    Runs the single-frame pipeline on the LAST frame of traj_file:
        native contacts -> binding/spike angles -> well-formed clusters

    Cluster size is reported in DIMERS, not segids/monomers -- each physical
    dimer is 2 segids (e.g. C5+D5), always added to a cluster as a complete
    pair via the get_partner union step, so len(segments) is always even and
    len(segments) // 2 is the true subunit (dimer) count.

    Returns (max_cluster_size_dimers, n_clusters, last_frame_index)
    """
    u = mda.Universe(pdb_file, traj_file)
    last_frame_idx = len(u.trajectory) - 1
    u.trajectory[last_frame_idx]

    contacts, computed_cutoffs = load_native_contacts(contact_dir)
    iface_bonds = last_frame_contacts(u, contacts, computed_cutoffs)
    interface_data = last_frame_angles(u, iface_bonds, contacts_per_bond)
    clusters = build_well_formed_clusters(iface_bonds, interface_data, contacts_per_bond,
                                          binding_angle_range, spike_angle_range)

    sizes_dimers = [len(c['segments']) // 2 for c in clusters]
    max_size_dimers = max(sizes_dimers) if sizes_dimers else 0
    return max_size_dimers, len(clusters), last_frame_idx


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pdb', default=f'{HBV_ENM_PATH}/scripts/important_oligomer_pdbs/abcd_capsid.pdb',
                   help='Topology PDB matching traj_file')
    p.add_argument('--traj', required=True,
                   help='Trajectory file (only the last frame is analyzed)')
    p.add_argument('--contactdir', default=f'{HBV_ENM_PATH}/scripts/claude_computed_contact_files',
                   help='Directory containing {A,B,C,D}_contacts_with_computed.txt')
    p.add_argument('--contacts', type=int, default=20,
                   help='Number of native contacts to be considered bonded')
    p.add_argument('--binding_range', default='0.0,1.5',
                   help="binding_angle well-formed range as 'min,max' (radians)")
    p.add_argument('--spike_range', default='0.0,1.0',
                   help="spike_angle well-formed range as 'min,max' (radians)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    binding_range = tuple(float(x) for x in args.binding_range.split(','))
    spike_range = tuple(float(x) for x in args.spike_range.split(','))

    max_size_dimers, n_clusters, last_frame_idx = analyze_last_frame(
        args.pdb, args.traj, args.contactdir, args.contacts,
        binding_angle_range=binding_range, spike_angle_range=spike_range,
    )

    print(f"Last frame:              {last_frame_idx}")
    print(f"Binding range:           {binding_range}")
    print(f"Spike range:             {spike_range}")
    print(f"n_clusters:              {n_clusters}")
    print(f"max_cluster_size_dimers: {max_size_dimers}")
