"""
full_analysis.py
Contains the functions for detecting bonds via native contacts

Input arguements:
    cutoff (in Angstroms)               -- default = 7Å
    path_to_pdb_file                    -- atom positions + bond topology
    path_to_trajectory_file             -- trajectory data
    {A,B,C,D}_contacts.txt folder       -- native contacts data
    output_directory_path               -- output directory

Output
    Main functions: parse_computed_cutoff, build_all_clusters,
    build_interface_angle_data, build_all_well_formed_clusters, and
    assign_persistent_cluster_ids

    parse_computed_cutoffs returns: 2 dictionaries containing native contact bond data:
    -  iface_contacts[frame_#][B1, C2] = (number of bonds in the B1-C2 interface in frame_#)
    -  type_data[frame_#]{B1 - C2 : ([resid1, resid2], [resid3, resid4]...}

    build_all_clusters returns: A dictionary containing cluster info per frame
    -   clusters[frame_#][list indexed by 0,1,2...]{segments: (all seg_ids
        in cluster), interfaces: (all interfaces in cluster)}

    build_interface_angle_data returns: A dictionary of binding/spike angles
    per interface per frame
    -   interface_data[frame_#][interface_name] = {binding_angle, spike_angle}

    build_all_well_formed_clusters returns: same shape as build_all_clusters,
    but interfaces only count as active if their binding_angle and
    spike_angle both fall within a fixed range
"""

import argparse
import os
import sys
import pickle
import numpy as np
import pandas as pd
import MDAnalysis as mda
import matplotlib.pyplot as plt
from collections import defaultdict
from MDAnalysis.lib.distances import capped_distance

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
    p.add_argument('--traj',       default=f'{HBV_ENM_PATH}/scripts/lammps_out/seg.dcd',
                   help='trajectory file for pdb')
    p.add_argument('--contactdir', default=f'{HBV_ENM_PATH}/scripts/claude_computed_contact_files',
                   help='Directory containing A_contacts.txt … D_contacts.txt')
    p.add_argument('--contacts',    type=int, default=19,
                   help='The number of contacts to be considered bonded')
    p.add_argument('--output_dir',     default='',
                   help=r'By default the pkl file is sent to traj_file dir' \
                   r'This argument adds a path: {output_dir}{traj_file_dir}')
    return p.parse_args()

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

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

def minimum_image(diff, box):
    """
    Applies the minimum-image convention to a displacement vector, same as
    the PBC correction used in parse_frames_computed_cutoffs. Without this,
    a displacement between two atoms that got wrapped into different
    periodic images (e.g. two residues of the same chain split across a
    box boundary) can come out with close to the wrong sign/direction.
    """
    return diff - box * np.round(diff / box)

# ---------------------------------------------------------------------------
# Bond Constuction Functions
# ---------------------------------------------------------------------------

# Right-column chain type for each contacts file (derived from spatial query exclusions)
#   A_contacts.txt : A chain contacts A chain
#   B_contacts.txt : B chain contacts C chain  (D excluded in original query)
#   C_contacts.txt : C chain contacts D chain  (A and B excluded, not-D{mn} leaves D other)
#   D_contacts.txt : D chain contacts B chain  (C and A excluded)
CONTACT_PARTNER_CHAIN = {'A': 'A', 'B': 'C', 'C': 'D', 'D': 'B'}

# Maps the monomer from one dimer to its other monomer pair
DIMER_MAP = {'A' : 'B', 'B' : 'A', 'C' : 'D', 'D' : 'C'}

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

    stdev_bond = 3 #Gaussian stdev in the histogram is about 2-3 angstroms

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
            cutoff = computed_cutoffs[(iface, res1, res2)] + 2 * stdev_bond
            pair_selections.append((iface, res1, res2, ag1, ag2, cutoff))

    skin          = 35.0  # 35 # Angstroms beyond cutoff used to build the neighbor list
    rebuild_every = 10   # rebuild neighbor list every N frames
    nl = {}              # idx -> array of [i, j] atom-index pairs within cutoff + skin

    for ts in u.trajectory:
        frame = ts.frame
        # print(f'Frame number:{frame}')

        # Rebuild neighbor list every rebuild_every frames
        if frame % rebuild_every == 0:
            #print(f"Rebuilding list, Frame #:{frame}")
            nl = {}
            for idx, (iface, res1, res2, ag1, ag2, cutoff) in enumerate(pair_selections):
                pairs = capped_distance(
                    ag1.positions, ag2.positions,
                    max_cutoff=cutoff + skin, box=u.dimensions, return_distances=False
                )
                if len(pairs) > 0:
                    nl[idx] = pairs

        # Only check atom pairs stored in the neighbor list
        box = u.dimensions[:3]
        for idx, (iface, res1, res2, ag1, ag2, cutoff) in enumerate(pair_selections):
            if idx not in nl:
                continue
            for (i, j) in nl[idx]:
                atom1, atom2 = ag1[i], ag2[j]
                if atom1.segid == atom2.segid:
                    continue
                diff = ag1.positions[i] - ag2.positions[j]
                diff -= box * np.round(diff / box)   # minimum image convention
                dist = np.sqrt(np.dot(diff, diff))
                if dist < cutoff:
                    iface_bonds[frame][(atom1.segid, atom2.segid)] += 1
                    iface_name = f"{atom1.segid}-{atom2.segid}"
                    iface_res_data[frame][iface_name][(res1, res2)] = dist

    return iface_bonds, iface_res_data


# ---------------------------------------------------------------------------
# Cluster Functions
# ---------------------------------------------------------------------------

def get_partner(residue_name):
    """
    Returns the monomer that is on the other side of the dimer
    """
    res_partner_chain = DIMER_MAP[residue_name[0]]
    res_partner = res_partner_chain + residue_name[1:]
    return res_partner

def build_all_clusters(iface_bonds, contacts_per_bond):
    """
    For each frame, finds connected components of monomers linked by
    enough contacts, and records which interfaces are active in each cluster.

    Returns:
        clusters[frame] = list of dicts, each with:
            'segids'   : frozenset of segids in the cluster
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

            # Adds on the respective monomer for the dimer pair
            s1partner_dimer = get_partner(s1)
            s2partner_dimer = get_partner(s2)
            neighbors[s1].add(s1partner_dimer)
            neighbors[s2].add(s2partner_dimer)

        # BFS to find connected components
        visited = set()
        frame_clusters = []
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
                # Set subtraction - removes all visited interfaces from the queue
                queue.extend(neighbors.get(curr, set()) - visited)

            # Collect all active edges whose both endpoints are in this component
            ifaces = {pair: count for pair, count in active_edges.items()
                      if ((pair[0] in component and pair[1] in component) and not (pair[1] == get_partner(pair[0])))}


            frame_clusters.append({
                'segids':   frozenset(component),
                'interfaces': ifaces,
            })
        
        # print(f"Frame_clusters: {frame_clusters}")
        # print(f"Frame number:{frame}  Number of clusters:{len(frame_clusters)}")
        clusters[frame] = frame_clusters

    return clusters


# ---------------------------------------------------------------------------
# Angle Functions
# ---------------------------------------------------------------------------

def build_interface_angle_data(pdb_file, traj_file, iface_bonds, contacts_per_bond):
    """
    For every frame, iterates over all active interfaces (edges in iface_bonds
    meeting contacts_per_bond) and computes both the binding angle (as in
    get_binding_angles.calculate_binding_angle) and the spike angle (as in
    get_spike_angle.calculate_spike_angle) at that interface.

    Atom selections for resid 132 (binding angle) and resid 73 / 58 (spike
    angle) are pre-built once per segid, and the trajectory is walked in a
    single forward pass, rather than re-selecting atoms and randomly seeking
    frames for every interface as calculate_binding_angle/calculate_spike_angle
    would if called directly per interface-frame pair.

    Returns:
        interface_data[frame][interface_name] = {
            'binding_angle': angle (radians),
            'spike_angle':   angle (radians),
        }
    """
    print("\nComputing binding + spike angles for all interfaces...\n")
    u = mda.Universe(pdb_file, traj_file)

    segids = set(u.select_atoms('name CA').segids)
    sel_132 = {seg: u.select_atoms(f'resid 132 and segid {seg}') for seg in segids}
    sel_73  = {seg: u.select_atoms(f'resid 73 and segid {seg}') for seg in segids}
    sel_58  = {seg: u.select_atoms(f'resid 58 and segid {seg}') for seg in segids}

    interface_data = defaultdict(dict)

    for ts in u.trajectory:
        frame = ts.frame
        frame_data = iface_bonds.get(frame, {})
        box = u.dimensions[:3]

        for (segid1, segid2), count in frame_data.items():
            if count < contacts_per_bond:
                continue
            partner1, partner2 = get_partner(segid1), get_partner(segid2)
            if any(s not in sel_132 for s in (segid1, segid2, partner1, partner2)):
                continue

            # Binding angle: dimer-axis direction of each monomer (self -> partner, resid 132)
            diff1_bind = minimum_image(sel_132[partner1].positions[0] - sel_132[segid1].positions[0], box)
            diff2_bind = minimum_image(sel_132[partner2].positions[0] - sel_132[segid2].positions[0], box)
            v1_bind = normalize_vector(diff1_bind)
            v2_bind = normalize_vector(diff2_bind)
            binding_angle = np.arccos(np.clip(dot_product(v1_bind, v2_bind), -1.0, 1.0))

            # Spike angle: spike direction of each monomer (resid 58 -> resid 73)
            diff1_spike = minimum_image(sel_73[segid1].positions[0] - sel_58[segid1].positions[0], box)
            diff2_spike = minimum_image(sel_73[segid2].positions[0] - sel_58[segid2].positions[0], box)
            v1_spike = normalize_vector(diff1_spike)
            v2_spike = normalize_vector(diff2_spike)
            spike_angle = np.arccos(np.clip(dot_product(v1_spike, v2_spike), -1.0, 1.0))

            interface_data[frame][f"{segid1}-{segid2}"] = {
                'binding_angle': binding_angle,
                'spike_angle':   spike_angle,
            }

    return interface_data


def build_all_well_formed_clusters(iface_bonds, interface_data, contacts_per_bond):
    """
    Same as build_all_clusters, but an edge only counts as active if, in
    addition to meeting the contact-count threshold, its binding angle and
    spike angle (looked up from interface_data, see build_interface_angle_data)
    both fall within a fixed range. This produces clusters built only out of
    "well-formed" interfaces.

    Returns the same structure as build_all_clusters:
        clusters[frame] = list of dicts, each with:
            'segids'   : frozenset of segids in the cluster
            'interfaces' : dict mapping (segid1, segid2) -> contact count
                           for every active edge within the cluster
    """

    # Parameterized against capsid spike and binding angle data
    BINDING_ANGLE_RANGE = (0.7, 1.5)  # radians -- tune to the desired binding angle window
    SPIKE_ANGLE_RANGE   = (0, 1.2)  # radians -- tune to the desired spike angle window

    print("\nBuilding well-formed clusters...\n")
    print(f"Binding Angle range: {BINDING_ANGLE_RANGE}")
    print(f"Spike Angle range: {SPIKE_ANGLE_RANGE}")
    clusters = {}

    for frame, frame_data in iface_bonds.items():
        frame_angle_data = interface_data.get(frame, {})

        # Collect active edges (meet the contact threshold AND both angle ranges)
        active_edges = {}
        for (s1, s2), count in frame_data.items():
            if count < contacts_per_bond:
                continue
            angles = frame_angle_data.get(f"{s1}-{s2}")
            if angles is None:
                continue
            if not (BINDING_ANGLE_RANGE[0] <= angles['binding_angle'] <= BINDING_ANGLE_RANGE[1]):
                continue
            if not (SPIKE_ANGLE_RANGE[0] <= angles['spike_angle'] <= SPIKE_ANGLE_RANGE[1]):
                continue
            active_edges[(s1, s2)] = count

        # Build adjacency from active edges
        neighbors = defaultdict(set)
        for (s1, s2) in active_edges:
            neighbors[s1].add(s2)
            neighbors[s2].add(s1)

            # Adds on the respective monomer for the dimer pair
            s1partner_dimer = get_partner(s1)
            s2partner_dimer = get_partner(s2)
            neighbors[s1].add(s1partner_dimer)
            neighbors[s2].add(s2partner_dimer)

        # BFS to find connected components
        visited = set()
        frame_clusters = []
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

            # Collect all active edges whose both endpoints are in this component
            ifaces = {pair: count for pair, count in active_edges.items()
                      if ((pair[0] in component and pair[1] in component) and not (pair[1] == get_partner(pair[0])))}

            frame_clusters.append({
                'segids':   frozenset(component),
                'interfaces': ifaces,
            })

        clusters[frame] = frame_clusters

    return clusters


def assign_persistent_cluster_ids(clusters, min_overlap=0.5):
    """
    Assigns a persistent 'cluster_id' to each cluster dict, in place, by
    matching it to a cluster in the previous frame with the highest segid
    overlap (Jaccard similarity of the 'segids' set). A cluster keeps its id
    across frames as long as enough of its segids persist; otherwise it is
    treated as newly formed and gets the next unused id.

    Works on either build_all_clusters or build_all_well_formed_clusters
    output -- call it separately on each, since a cluster splitting under
    the well-formed threshold is a distinct identity from the raw cluster.

    Frames are processed in sorted (chronological) order regardless of the
    dict's iteration order.

    Returns clusters (same object, mutated in place, for chaining).
    """
    next_id = 0
    prev_frame_clusters = []  # list of (cluster_id, segids)

    for frame in sorted(clusters):
        frame_clusters = clusters[frame]
        used_ids = set()

        for cluster in frame_clusters:
            segids = cluster['segids']
            best_id, best_score = None, 0.0

            for cid, prev_segids in prev_frame_clusters:
                if cid in used_ids:
                    continue
                overlap = len(segids & prev_segids) / len(segids | prev_segids)
                if overlap > best_score:
                    best_id, best_score = cid, overlap

            if best_id is not None and best_score >= min_overlap:
                chosen_id = best_id
            else:
                chosen_id = next_id
                next_id += 1

            used_ids.add(chosen_id)
            cluster['cluster_id'] = chosen_id

        prev_frame_clusters = [(c['cluster_id'], c['segids']) for c in frame_clusters]

    return clusters


# ---------------------------------------------------------------------------
# Saving Data Functions
# ---------------------------------------------------------------------------

def save_cluster_data(pdb_file, traj_file, contact_dir, contacts_per_bond, output_dir):
    """
    Runs the full pipeline:
        1. parse_frames_computed_cutoffs   -> iface_bonds, iface_res_data
        2. build_all_clusters              -> all_clusters
        3. build_interface_angle_data      -> interface_data
        4. build_all_well_formed_clusters  -> all_well_formed_clusters

    Saves all five to a pickle file at cluster_data.pkl as:
        {
            'iface_bonds':              iface_bonds,
            'iface_res_data':           iface_res_data,
            'all_clusters':             all_clusters,
            'interface_data':           interface_data,
            'all_well_formed_clusters': all_well_formed_clusters,
            'pdb_file':                 pdb_file,
            'traj_file':                traj_file,
        }

    Returns the same five objects.
    """
    iface_bonds, iface_res_data = parse_frames_computed_cutoffs(pdb_file, traj_file, contact_dir)
    all_clusters = build_all_clusters(iface_bonds, contacts_per_bond)
    assign_persistent_cluster_ids(all_clusters)
    interface_data = build_interface_angle_data(pdb_file, traj_file, iface_bonds, contacts_per_bond)
    all_well_formed_clusters = build_all_well_formed_clusters(iface_bonds, interface_data, contacts_per_bond)
    assign_persistent_cluster_ids(all_well_formed_clusters)

    print("\nSaving cluster data...\n")
    output_path = f"{output_dir}{traj_file}"
    output_path = output_path.replace("seg.dcd", "complete_cluster_data.pkl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({
            # Only the outer defaultdict needs stripping (its factory is a
            # lambda, which pickle can't serialize) -- the inner
            # defaultdict(int)/defaultdict(dict) levels are already picklable
            # since their factories are plain builtins. A shallow dict(...)
            # avoids deep-copying the entire (multi-GB, for a full capsid)
            # nested structure a second time right before pickling.
            'iface_bonds':              dict(iface_bonds),
            'iface_res_data':           dict(iface_res_data),
            'all_clusters':             all_clusters,
            'all_well_formed_clusters': all_well_formed_clusters,
            'interface_data':           dict(interface_data),
            'pdb_file':                 pdb_file,
            'traj_file':                traj_file,
        }, f)
    print(f"Saved cluster data to {output_path}")

    return iface_bonds, iface_res_data, all_clusters, interface_data, all_well_formed_clusters


if __name__ == "__main__":

    args = parse_args()
    iface_bonds, iface_res_data, all_clusters, interface_data, all_well_formed_clusters = save_cluster_data(
        args.pdb, args.traj, args.contactdir, args.contacts, args.output_dir
    )
