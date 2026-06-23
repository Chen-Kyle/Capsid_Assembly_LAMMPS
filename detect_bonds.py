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
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")

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
    -  iface_contacts[frame_#][B1, C2] = (number of bonds in the B1-C2 interface in frame_#)
    -  type_data[frame_#]{B1 - C2 : ([resid1, resid2], [resid3, resid4]...}
    """
    iface_contacts = defaultdict(lambda: defaultdict(int))
    type_data = defaultdict(lambda: defaultdict(list))

    u = mda.Universe(pdb_file, traj_file)
    
    contacts = build_native_contacts(contact_dir)
    Nframes = len(u.trajectory)
    print(Nframes)

    # Pre-compute all selections once
    pair_selections = []
    for iface, pairs in contacts.items():
        for (chain1, res1, chain2, res2) in pairs:
            ag1 = u.select_atoms(f'resid {res1}')
            ag2 = u.select_atoms(f'resid {res2}')
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
                        iface_contacts[frame][(atom1.segid, atom2.segid)] += 1
                        iface_name = f"{atom1.segid}-{atom2.segid}"
                        type_data[frame][iface_name].append((res1, res2))

    return iface_contacts, type_data

    # # Parses through every frame for every contact present in every interface
    # for frame in range(Nframes):
    #     print(f'Frame: {frame}')
    #     u.trajectory[frame]
    #     for iface, pairs in contacts.items():
    #         for (chain1, res1, chain2, res2) in pairs:
    #             atoms1 = u.select_atoms(f'resnum {res1}')
    #             atoms2 = u.select_atoms(f'resnum {res2}')
    #             for atom1 in atoms1:
    #                 pos1 = atom1.position
    #                 chain_id1 = atom1.segid
    #                 for atom2 in atoms2:
    #                     pos2 = atom2.position
    #                     chain_id2 = atom2.segid
    #                     distance = mda.lib.distances.calc_bonds(pos1, pos2, box=u.dimensions)
    #                     if (distance < cutoff and chain_id1 != chain_id2):
    #                         iface_contacts[frame][(chain_id1, chain_id2)] += 1
    #                         type_data[frame][iface].append((res1, res2))


if __name__ == "__main__":
    pdb_file = "/home/kyle/2026_Research/HBV_enm/important_oligomer_pdbs/cg_ABCD_separate.pdb"
    traj_file = "/home/kyle/2026_Research/trajectory_files/ABCD_seg.dcd"
    contact_dir = "/home/kyle/2026_Research/HBV_enm/contact_files"
    iface_contacts, type_data = parse_frames(7, pdb_file, traj_file, contact_dir)
    print(iface_contacts)
    print(type_data)