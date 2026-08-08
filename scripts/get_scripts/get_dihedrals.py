"""
Input arguements:
    path_to_pkl_file                    -- all cluster information

Output
    A csv file containing the dihedral angles which displays the data like so:
    Frame#, Dimer, Dihedral Angle
"""

import argparse
import os
import MDAnalysis as mda
import numpy as np
import pickle

import warnings
warnings.filterwarnings("ignore")

HBV_ENM_PATH = os.environ.get("HBV_ENM_PATH", "/home/kyle/2026_Research/HBV_enm")

# Maps the monomer from one dimer to its other monomer pair
PARTNER_MAP = {'A' : 'B', 'B' : 'A', 'C' : 'D', 'D' : 'C'}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pkl',
                   help='Path to the trajectory to analyze', default=f'{HBV_ENM_PATH}/scripts/lammps_out/complete_cluster_data.pkl')
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
    segIDs = interface_name.split('-')
    return segIDs

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

def minimum_image(diff, box):
    """
    Applies the minimum-image convention to a displacement vector. Without
    this, a displacement between two atoms that got wrapped into different
    periodic images (e.g. two chains split across a box boundary) can come
    out with close to the wrong sign/direction.
    """
    return diff - box * np.round(diff / box)

def detect_dimer_list(u_sim):
    """
    Auto-detect which AB and CD dimers are present in the PDB from its segIDs.
    Returns a list like ['A1B1', 'C1D1', 'A2B2', ...] ordered by monomer number,
    interleaved AB/CD as in the original decamer setup.
    """
    segids = set(u_sim.select_atoms('name CA').segids)
    segidNums = set(int(s[1:]) for s in segids)
    n_dimers = max(segidNums) + 1
    dimer_list = []
    for i in range(1, n_dimers):
        if f'A{i}' in segids and f'B{i}' in segids:
            dimer_list.append(f'A{i}-B{i}')
        if f'C{i}' in segids and f'D{i}' in segids:
            dimer_list.append(f'C{i}-D{i}')
    return dimer_list

def remove_segid_from_interface_list(segid_to_remove, interface_list):
    """
    Removes a segid from a list of interfaces.
    An interface list looks like:
    interfaces_containing_segid1: [('A1', 'A5'), ('A2', 'A1')]
    ->
    interfaces_containing_segid1: [A5, A2]
    """
    segid_list = []
    for interface in interface_list:
        for segid in interface:
            if segid != segid_to_remove:
                segid_list.append(segid)
    return segid_list


# ---------------------------------------------------------------------------
# Core Functions for dihedral calculation
# ---------------------------------------------------------------------------

def check_valid(dimer, well_formed_clusters_frame):
    """
    Checks to see if the dihedral can be calculated by looking at the active edges
    in well-formed clusters. 

    If it can, it will return a list of interfaces in the dihedral. STARTING with
    the interfaces on each tip of the diamond of the dihedral followed by the rest of the interfaces
    
    ex. if the dimer is A1-B1: 
    [('B5', 'C5'), ('B2', 'D1'), ... ('A1', 'A5')]

    AND a boolean value declaring the dihedral valid or not
    """
    dihedral_status = False
    dihedral_interfaces = []
    segid1, segid2 = get_segids(dimer)

    for cluster in well_formed_clusters_frame:
        if segid1 in cluster['segids'] and segid2 in cluster['segids']:
            # Finds the interfaces that involve the dimer
            interfaces_containing_segid1 = [interface_containing_segid1 for interface_containing_segid1 in cluster['interfaces'] if segid1 in interface_containing_segid1]
            interfaces_containing_segid2 = [interface_containing_segid2 for interface_containing_segid2 in cluster['interfaces'] if segid2 in interface_containing_segid2]

            # Isolates the segids that are in contact with the respective segid of the dimer
            segids_in_contact_with_segid1 = remove_segid_from_interface_list(segid1, interfaces_containing_segid1)
            segids_in_contact_with_segid2 = remove_segid_from_interface_list(segid2, interfaces_containing_segid2)

            # Checks to see if an interface exists in the cluster which connects the segids in contact with the dimer
            for segid_in_contact_with_segid1 in segids_in_contact_with_segid1:
                partner1 = get_partner(segid_in_contact_with_segid1)
                for segid_in_contact_with_segid2 in segids_in_contact_with_segid2:
                    partner2 = get_partner(segid_in_contact_with_segid2)
                    if (partner1, partner2) in cluster['interfaces']:
                        dihedral_interfaces.append([partner1, partner2])
                    elif (partner2, partner1) in cluster['interfaces']:
                        dihedral_interfaces.append([partner2, partner1])
    
    if len(dihedral_interfaces) == 2:
        for interface in interfaces_containing_segid1:
            dihedral_interfaces.append(list(interface))
        for interface in interfaces_containing_segid2:
            dihedral_interfaces.append(list(interface))

        dihedral_status = True
    return dihedral_interfaces, dihedral_status


def check_still_valid(dimer, well_formed_clusters_frame, dimer_dihedral_list):
    """
    Checks to see if all of the interfaces in the dihedral still exist in the
    current frame
    
    Then returns the same interface list if true or returns an empty list if false
    while also updating the dihedral status:
    
    {partner_interfaces}, dihedral_status
    """
    interfaces_to_check = dimer_dihedral_list[0]

    # Collect all active interfaces across every cluster in this frame
    all_active = set()
    for cluster in well_formed_clusters_frame:
        for iface in cluster['interfaces']:
            all_active.add(iface)

    # If any interface is missing from the active set, the dihedral is no longer valid
    if any(tuple(iface) not in all_active for iface in interfaces_to_check):
        return [], False

    return interfaces_to_check, True

def calculate_dihedral(dihedral_interfaces, atom_selections, box_dims):
    """
    Calculates the dihedral angle given the interfaces
    """

    tip_interfaces = dihedral_interfaces[:2]
    cross_products = []
    for interface in tip_interfaces:
        distance_vectors = []
        for segid in interface:
            partner = get_partner(segid)
            distance_vector = minimum_image(atom_selections[partner].positions[0] - atom_selections[segid].positions[0], box_dims)
            normalized_vector = normalize_vector(distance_vector)
            distance_vectors.append(normalized_vector)

        cross_products.append(cross_product(distance_vectors[0], distance_vectors[1]))

    dp = dot_product(cross_products[0], cross_products[1])
    dihedral_angle = np.arccos(dp)
    return dihedral_angle

def build_dihedral_data(u, pkl_data):
    """
    Parses through every frame while checking to see if a dihedral calculation is possible
    Then if it can, it will calculate the dihedral and save it to dihedral_data
    """

    Nframes = len(u.trajectory)
    dihedral_data = {}

    print("\nBuilding Dihedral Data\n")
    print(f"\nLength of trajectory: {Nframes}")

    # Builds the atom groups and stores them to a dictionary with the vertex
    # resid 132 stored as the first atom group
    atom_selections = {}
    segids = set(u.select_atoms('name CA').segids)
    for segid in segids:
        atom_selections[segid] = u.select_atoms(f'resid 132 and segid {segid}')

    # Construct dimer list
    dimer_list = detect_dimer_list(u)

    # Loops through the simulation
    well_formed_clusters = pkl_data['all_well_formed_clusters']
    box_dims = u.dimensions[:3]

    for timestep in u.trajectory:
        frame = timestep.frame

        dihedrals_list = {}
        dihedral_data[frame] = {}
        for dimer in dimer_list:
            # Checks to see if the dihedral can still be calculated
            if dimer not in dihedrals_list or dihedrals_list[dimer][1] == False:
                dihedrals_list[dimer] = check_valid(dimer, well_formed_clusters[frame])
            else:
                dihedrals_list[dimer] = check_still_valid(dimer, well_formed_clusters[frame], dihedrals_list[dimer])

            # Calculates the dihedral if the status is True
            if dihedrals_list[dimer][1]:
                dihedral_data[frame][dimer] = calculate_dihedral(dihedrals_list[dimer][0], atom_selections, box_dims)

    return dihedral_data


def print_dihedral_angle_data(dihedral_data, pkl_file_path):
    """
    Prints out the binding angle data to the same directory as the pkl file
    Uses the format:{Frame#}   {Dimer}   {Dihedral_angle}
    """
    print("Printing Dihedral Angles...\n")
    out_file_name = pkl_file_path.replace('complete_cluster_data.pkl', 'dihedral_angles.csv')
    with open(out_file_name, "w") as f:
        # Header
        f.write("Frame#,Dimer,Dihedral Angle\n")

        for frame, dimer_dihedral_angles in dihedral_data.items():
            for dimer, dihedral_angle in dimer_dihedral_angles.items():
                f.write(f"{frame},{dimer},{dihedral_angle:.6f}\n")

    print(f"Saved to: {out_file_name}")
    return


# ---------------------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------------------

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

    dihedral_data = build_dihedral_data(u, pkl_data)
    print_dihedral_angle_data(dihedral_data, args.pkl)