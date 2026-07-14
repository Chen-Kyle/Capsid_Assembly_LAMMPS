"""
get_distances.py
Must be used between two dimers (typically sharing an interface)
WARNING!!!!: for A-A interfaces make sure to test both A1-A2 as well as A2-A1
if you are getting questionable results because this script will not get the lowest one

Input arguements:
    dimer_names                         -- default = A2-A1
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
import matplotlib.pyplot as plt
import pandas as pd
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.lib.distances import calc_bonds

import warnings
warnings.filterwarnings("ignore")


HBV_ENM_PATH= os.environ.get("HBV_ENM_PATH", "/home/kyle/2026_Research/HBV_enm")

# Right-column chain type for each contacts file (derived from spatial query exclusions)
#   A_contacts.txt : A chain contacts A chain
#   B_contacts.txt : B chain contacts C chain  (D excluded in original query)
#   C_contacts.txt : C chain contacts D chain  (A and B excluded, not-D{mn} leaves D other)
#   D_contacts.txt : D chain contacts B chain  (C and A excluded)
CONTACT_PARTNER_CHAIN = {'A': 'A', 'B': 'C', 'C': 'D', 'D': 'B'}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dnames',
                   help='Names of the dimers in the interface. ex: A1-A2', default='A2-A1')
    p.add_argument('--pdb',
                   help='Path to the pdb file', default=f'{HBV_ENM_PATH}/scripts/important_oligomer_pdbs/cg_AA_avg.pdb')
    p.add_argument('--traj',
                   help='Path to the trajectory to analyze', default=f'/home/kyle/storage/kyle_storage/HBV_enm/trajectory_files/distogram_trajs/cg_A1A2_avg.dcd')
    p.add_argument('--out',
                   help='Path to the output HDF5 file', default=f'{HBV_ENM_PATH}/scripts/')
    p.add_argument('--contact_dir',
                   help='Path to directory with all the native contact information',  default=f'{HBV_ENM_PATH}/scripts/contact_files')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_dnames(dnames):
    dnames_list = dnames.split("-")
    print(dnames_list)
    if dnames_list[0][1].isdigit() and dnames_list[1][1].isdigit():
        print(f"dnames found: {dnames_list[0]}, {dnames_list[1]}")
        return dnames_list
    else:
        raise ValueError("dnames input formatted incorrectly. Use a dash between the dimer names ex: B1-C1")

def build_native_contacts(contactdir):
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
        clist = pd.read_csv(f"{contactdir}/{iface}_contacts.txt",
                            sep=r'\s+', header=None, names=cols)
        for _, row in clist.iterrows():
            contacts[iface].append(
                (iface, int(row['resnum1']), partner, int(row['resnum2']))
            )
    return contacts

# ---------------------------------------------------------------------------
# Distances Calculator
# ---------------------------------------------------------------------------

def get_all_distances(dnames, pdb_file, traj_file, out_path):
    """
    Loops through all the frames in the trajectory and gets the distances
    between all of the different residues between two dimers in an interface,
    streaming each frame's distance matrix directly to an HDF5 file on disk
    instead of accumulating everything in memory (trajectories can be
    hundreds of thousands of frames long).

    Writes an HDF5 file at out_path containing:
    - 'distances'      : (nframes, n1, n2) array, distances per frame
    - 'mean_distances' : (n1, n2) array, mean distance across all frames

    Returns out_path.
    """

    u = mda.Universe(pdb_file, traj_file)
    nframes = len(u.trajectory)

    # Seperates them into the two different interfaces
    dname1, dname2 = get_dnames(dnames)
    chain1 = u.select_atoms(f'segid {dname1}')
    chain2 = u.select_atoms(f'segid {dname2}')
    chain1_len = int(len(chain1.residues))
    chain2_len = int(len(chain2.residues))
    print(f"chain1_len: {chain1_len} chain2_len: {chain2_len}")

    print(f"Parsing {nframes} frames to gather res pair distances")
    outfile = f'{out_path}_all_distances.h5'
    with h5py.File(outfile, 'w') as f:
        dset = f.create_dataset(
            'distances', shape=(nframes, chain1_len, chain2_len), dtype='float32')
        running_sum = np.zeros((chain1_len, chain2_len), dtype=np.float64)

        for ts in u.trajectory:
            frameN = ts.frame
            print(f'FrameN: {frameN}')
            pos1 = np.array([res.atoms.center_of_mass() for res in chain1.residues])
            pos2 = np.array([res.atoms.center_of_mass() for res in chain2.residues])

            distances = distance_array(pos1, pos2, box=u.dimensions).astype(np.float32)
            dset[frameN] = distances
            running_sum += distances

        f.create_dataset('mean_distances', data=running_sum / nframes)

    print(f"hd5py file created at: {outfile}")
    return outfile

def get_native_contact_distances(dnames, pdb_file, traj_file, out_path, contact_dir):
    """
    Does the same as get_all_distances, but restricted to only the residues
    that participate in native contacts for this dimer pair, rather than
    every residue in each chain.

    Writes an HDF5 file at out_path containing:
    - 'distances'      : (nframes, n1, n2) array, distances per frame
    - 'mean_distances' : (n1, n2) array, mean distance across all frames
    - 'resids1'/'resids2' : resids making up the rows/columns of the grid

    Returns out_path.
    """
    dimer1, dimer2 = get_dnames(dnames)
    u = mda.Universe(pdb_file, traj_file)
    nframes = len(u.trajectory)

    contacts = build_native_contacts(contact_dir)

    # Collects the unique resids (on each dimer) that show up in any native contact
    resids1 = set()
    resids2 = set()
    for _, pairs in contacts.items():
        for (chain1, res1, chain2, res2) in pairs:
            if dimer1[0] == chain1 and dimer2[0] == chain2:
                resids1.add(res1)
                resids2.add(res2)
            elif dimer2[0] == chain1 and dimer1[0] == chain2:
                resids1.add(res2)
                resids2.add(res1)

    resid1_sel = ' '.join(str(r) for r in sorted(resids1))
    resid2_sel = ' '.join(str(r) for r in sorted(resids2))

    chain1 = u.select_atoms(f'segid {dimer1} and resid {resid1_sel}')
    chain2 = u.select_atoms(f'segid {dimer2} and resid {resid2_sel}')
    chain1_len = int(len(chain1.residues))
    chain2_len = int(len(chain2.residues))
    print(f"native contact residues - dimer1: {chain1_len} dimer2: {chain2_len}")

    outfile = f'{out_path}_native_contacts_distances.h5'
    with h5py.File(outfile, 'w') as f:
        dset = f.create_dataset(
            'distances', shape=(nframes, chain1_len, chain2_len), dtype='float32')
        running_sum = np.zeros((chain1_len, chain2_len), dtype=np.float64)

        print(f"Parsing {nframes} frames to gather native contact residue distances")
        for ts in u.trajectory:
            frameN = ts.frame
            print(f"FrameN: {frameN}")
            pos1 = np.array([res.atoms.center_of_mass() for res in chain1.residues])
            pos2 = np.array([res.atoms.center_of_mass() for res in chain2.residues])

            distances = distance_array(pos1, pos2, box=u.dimensions).astype(np.float32)
            dset[frameN] = distances
            running_sum += distances

        f.create_dataset('mean_distances', data=running_sum / nframes)
        f.create_dataset('resids1', data=np.array([res.resid for res in chain1.residues]))
        f.create_dataset('resids2', data=np.array([res.resid for res in chain2.residues]))
    print(f"hd5py file created at: {outfile}")
    return outfile


# ---------------------------------------------------------------------------
# Plots the distograph using distance_data
# ---------------------------------------------------------------------------

def plot_distance_data(dname, h5_path):
    """
    Plots the mean distance across all of the frames for all of the residue pairs,
    reading only the small precomputed mean matrix from the HDF5 file (not the
    full per-frame dataset).

    If the file also has 'resids1'/'resids2' datasets (written by
    get_native_contact_distances), the axis ticks are labeled with the actual
    residue numbers instead of plain array index.
    """

    with h5py.File(h5_path, 'r') as f:
        distance_data_mean = f['mean_distances'][:]
        resids1 = f['resids1'][:] if 'resids1' in f else None
        resids2 = f['resids2'][:] if 'resids2' in f else None

    fig, ax = plt.subplots()

    im = ax.imshow(distance_data_mean, cmap='viridis', origin='lower')#, vmax=50)
    fig.colorbar(im, ax=ax, label='Distance (Å)')

    dname1, dname2 = get_dnames(dname)

    ax.set_xlabel(f'Residue {dname2}')
    ax.set_ylabel(f'Residue {dname1}')

    if resids1 is not None and resids2 is not None:
        ax.set_yticks(range(len(resids1)))
        ax.set_yticklabels(resids1)
        ax.set_xticks(range(len(resids2)))
        ax.set_xticklabels(resids2, rotation=90)
        plt.savefig(f'{dname}_native_contacts_distogram.png', dpi=300)
    else:
        plt.savefig(f'{dname}_distogram.png', dpi=300)
        
    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    # outfile = get_native_contact_distances(args.dnames, args.pdb, args.traj, args.out, args.contact_dir)
    outfile = get_all_distances(args.dnames, args.pdb, args.traj, args.out)
    
    # outfile = "scripts/distances.h5"
    plot_distance_data(args.dnames, outfile)

    outfile = get_native_contact_distances(args.dnames, args.pdb, args.traj, args.out, args.contact_dir)
    # outfile = get_all_distances(args.dnames, args.pdb, args.traj, args.out)
    
    # outfile = "scripts/distances.h5"
    plot_distance_data(args.dnames, outfile)