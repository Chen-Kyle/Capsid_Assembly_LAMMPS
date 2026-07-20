"""
get_dihedrals.py
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
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.lib.distances import calc_bonds

import warnings
warnings.filterwarnings("ignore")

CONTACTS_PER_BOND = 10
HBV_ENM_PATH = os.environ.get("HBV_ENM_PATH", "/home/kyle/2026_Research/HBV_enm")

# Maps the monomer from one dimer to its other monomer pair
PARTNER_MAP = {'A' : 'B', 'B' : 'A', 'C' : 'D', 'D' : 'C'}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dimer',
                   help='Dimer name to check the dihedral of ie: A1-B1', default=f'C1-D1')
    p.add_argument('--pdb',
                   help='Path to the pdb file', default=f'{HBV_ENM_PATH}/scripts/important_oligomer_pdbs/pentamer_avg.pdb')
    p.add_argument('--traj',
                   help='Path to the trajectory to analyze', default=f'{HBV_ENM_PATH}/scripts/lammps_out/seg.dcd')
    p.add_argument('--pkl',
                   help='Path to the trajectory to analyze', default=f'{HBV_ENM_PATH}/scripts/lammps_out/seg_cluster_data.pkl')
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

def get_chains(interface_name):
    """
    Extracts the chainIDs out of the interface name string
    """
    chains = interface_name.split('-')
    return chains

def cross_product(vector1, vector2):
    """
    Takes the cross product of two 3D vectors: vector1 x vector2
    """
    x_component = vector1[1]*vector2[2] - vector2[1]*vector1[2]
    y_component = vector1[2]*vector2[0] - vector2[2]*vector1[0]
    z_component = vector1[0]*vector2[1] - vector2[0]*vector1[1]
    cross_product = x_component, y_component, z_component
    return cross_product

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

def find_partner_dimers(pdb_file, traj_file, dimer_name, pkl_data):
    """
    Finds the dimers that are connected to the one given to find the dihedral angles of the inputted dimer

    Returns:
        partner_dimers[cross_interface] = the interface on the opposite side of the triangle (only 2) with:
            [dimer1, dimer2] :  The other two dimers making up the triangle. 
                                Necessary for calculating the normal vector needed for the dihedral
    """
    u = mda.Universe(pdb_file, traj_file)
    iface_bonds = pkl_data['iface_bonds']
    
    frame_data = iface_bonds[0]
    # Collect active edges (meet the threshold) for frame 0
    active_edges = {pair: count for pair, count in frame_data.items()
                    if count >= CONTACTS_PER_BOND}

    chainID1, chainID2 = get_chains(dimer_name)
    dimer1_active_edges = {pair: count for pair, count in active_edges.items()
                            if chainID1 in pair}
    dimer2_active_edges = {pair: count for pair, count in active_edges.items()
                            if chainID2 in pair}

    # Loops through active edges of the dimer the dihedral is being calculated for
    # Then loops through the active edges of those active edges and finds the match
    partner_dimers = {}
    for interface1 in dimer1_active_edges.keys():
        # Makes sure not to check the dimer inputted
        for edge1 in interface1:
            if edge1 == chainID1:
                continue
            for interface2 in dimer2_active_edges.keys():
                for edge2 in interface2:
                    if edge2 == chainID2:
                        continue
                    # Get partner segid
                    partner1 = get_partner(edge1)
                    partner2 = get_partner(edge2)

                    partner1_active_edges = {pair: count for pair, count in active_edges.items()
                                            if partner1 in pair}
                    partner2_active_edges = {pair: count for pair, count in active_edges.items()
                                            if partner2 in pair}               
                    for edges in partner1_active_edges.keys():
                        if edges in partner2_active_edges.keys():
                            print(f"Valid set of dimers found.\nOne triangle is {chainID1}-{chainID2}, {edge1}-{partner1}, {edge2}-{partner2}")
                            triangle_end_dimers = []
                            triangle_end_dimers.append(f'{edge1}-{partner1}')
                            triangle_end_dimers.append(f'{edge2}-{partner2}')
                            partner_dimers[f'{partner1}-{partner2}'] = triangle_end_dimers
    
    print(f'partner_dimers: {partner_dimers}')
    return partner_dimers


# ---------------------------------------------------------------------------
# Dihedral Calculation
# ---------------------------------------------------------------------------

def calculate_dihedral(pdb_file, traj_file, partner_dimers, outpath, pkl_data):
    """
    Calculates the dihedral angles for every frame and saves it to an outfile:
        {outpath}/{dimer_name}_dihedrals.txt
    """

    u = mda.Universe(pdb_file, traj_file)
    Nframes = len(u.trajectory)
    dihedral_data = np.zeros(Nframes)

    print("\nCalculating Dihedrals\n")
    print(f"\nLength of trajectory: {Nframes}")

    # Builds the atom groups and stores them to a dictionary with the vertex
    # resid 132 stored as the first atom group
    atom_groups = {}
    atom_groups_debug = {}
    for main_vertex in partner_dimers:
        main_vertex_chains = get_chains(main_vertex)
        print(main_vertex_chains)

        for vertex_chain in main_vertex_chains:
            partner_chain = get_partner(vertex_chain)
            vertex_atom = u.select_atoms(f'resid 132 and segid {vertex_chain}')
            vertex_atom_partner = u.select_atoms(f'resid 132 and segid {partner_chain}')

            atom_groups[vertex_chain] = vertex_atom, vertex_atom_partner

    # Should try introducing some testing logic that monitors active_edges and then fails when any of the interfaces disappears from active_edges in any frame
    for ts in u.trajectory:
        frame = ts.frame
        frame_data = pkl_data['iface_bonds'][frame]
        active_edges = {pair: count for pair, count in frame_data.items()
                if count >= CONTACTS_PER_BOND}
        
        base_vectors = []
        for main_vertex in partner_dimers.keys():
            distance_vectors = []
            
            vertex_chains = get_chains(main_vertex)
            for vertex_chain in vertex_chains:
                # Gets the vectors for the cross product
                main_vertex_res132_pos = atom_groups[vertex_chain][0].positions[0]
                partner_res132_pos = atom_groups[vertex_chain][1].positions[0]
                distance_vectors.append(partner_res132_pos - main_vertex_res132_pos)

            # Takes the cross product of the distance vectors
            # Then gets the dot product and arccos of that to get the dihedral
            base_vector = cross_product(distance_vectors[0], distance_vectors[1])
            normalized_base_vector =  normalize_vector(base_vector)
            base_vectors.append(normalized_base_vector)
            
        dotted = dot_product(base_vectors[0], base_vectors[1])
        dihedral = np.arccos(abs(dotted))

        print(dihedral)

        dihedral_data[frame] = dihedral

    return dihedral_data

def plot_dihedrals(dihedral_data):
    plt.hist(dihedral_data, bins=30)
    plt.show()

if __name__ == '__main__':
    args = parse_args()
    pkl_data = load_pickle(args.pkl)
    partner_dimers = find_partner_dimers(args.pdb, args.traj, args.dimer, pkl_data)
    dihedral_data = calculate_dihedral(args.pdb, args.traj, partner_dimers, args.out, pkl_data)
    plot_dihedrals(dihedral_data)