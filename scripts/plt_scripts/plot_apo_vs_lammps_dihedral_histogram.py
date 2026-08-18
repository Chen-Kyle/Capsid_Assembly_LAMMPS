"""
plot_apo_vs_lammps_dihedral_histogram.py

Overlays four dihedral-angle distributions on one relative-frequency histogram:
  - apo_freeCp5_dihedralAB.txt                              (blue)
  - apo_freeCp5_dihedralCD.txt                              (lightblue)
  - LAMMPS pentamer_AB run's cluster_data.pkl, AB-group     (orange)
  - LAMMPS pentamer_CD run's cluster_data.pkl, CD-group     (moccasin / light orange)

The LAMMPS pkl files only store iface_bonds/iface_res_data/clusters, not the
dihedral itself, so this script reuses get_dihedrals.py's dimer-detection
and dihedral-calculation logic (same as its `--all` mode) to recompute the
per-frame, group-averaged dihedral time series before histogramming it.
"""

import argparse
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda

HBV_ENM_PATH = os.environ.get("HBV_ENM_PATH", "/home/kyle/2026_Research/HBV_enm")
SCRIPTS_DIR = f"{HBV_ENM_PATH}/scripts"
sys.path.insert(0, SCRIPTS_DIR)

import get_dihedrals as gd  # noqa: E402  (reuses detect_dimer_list/get_dimer_group/find_partner_dimers/calculate_dihedral)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ab_txt', default=f'{HBV_ENM_PATH}/data_from_carolina/apo_freeCp5_dihedralAB.txt',
                    help='Path to the apo AB dihedral text file')
    p.add_argument('--cd_txt', default=f'{HBV_ENM_PATH}/data_from_carolina/apo_freeCp5_dihedralCD.txt',
                    help='Path to the apo CD dihedral text file')
    p.add_argument('--ab_pkl',
                    default='/home/kyle/storage/kyle_storage/HBV_enm/trajectory_files/dihedral_trajs/Enative=0.5_pentamer_AB/10739697/cluster_data.pkl',
                    help='cluster_data.pkl from the LAMMPS pentamer_AB run')
    p.add_argument('--cd_pkl',
                    default='/home/kyle/storage/kyle_storage/HBV_enm/trajectory_files/dihedral_trajs/Enative=0.5_pentamer_CD/10739696/cluster_data.pkl',
                    help='cluster_data.pkl from the LAMMPS pentamer_CD run')
    p.add_argument('--out', default=f'{HBV_ENM_PATH}/apo_vs_lammps_dihedral_histogram.png')
    p.add_argument('--bins', type=int, default=50)
    return p.parse_args()


def load_txt_dihedral(path):
    return np.loadtxt(path)


def compute_lammps_group_dihedral(pkl_path, group):
    """
    Recomputes the per-frame dihedral angle, averaged across every valid
    dimer of the requested chain group ('AB' or 'CD'), for one LAMMPS run.
    Mirrors the `--all` branch of get_dihedrals.py's __main__.
    """
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)

    pdb_path = f'{SCRIPTS_DIR}/{pkl_data["pdb_file"]}'
    # pkl_data['traj_file'] points at a stale scratch path — the real
    # trajectory lives next to the pkl itself.
    traj_path = os.path.join(os.path.dirname(pkl_path), 'seg.dcd')

    print(f'Loading {pdb_path} + {traj_path} ...')
    u = mda.Universe(pdb_path, traj_path)

    dimers = gd.detect_dimer_list(u)
    group_dihedrals = []
    for dimer in dimers:
        if gd.get_dimer_group(dimer) != group:
            continue
        partner_dimers = gd.find_partner_dimers(dimer, pkl_data)
        if len(partner_dimers) != 2:
            print(f'------Dihedral cannot be calculated for: {dimer}------')
            continue
        group_dihedrals.append(gd.calculate_dihedral(u, partner_dimers))

    if not group_dihedrals:
        raise ValueError(f'No valid {group} dimers with a computable dihedral found in {pkl_path}')

    return np.mean(np.array(group_dihedrals), axis=0)


def plot_histogram(ab_txt, cd_txt, ab_pkl_data, cd_pkl_data, out_path, bins):
    series = [
        ('AB atom-model',    ab_txt,      'blue'),
        ('CD atom-model',    cd_txt,      'lightblue'),
        ('AB residue-model', ab_pkl_data, 'orange'),
        ('CD residue-model', cd_pkl_data, 'moccasin'),
    ]

    fig, ax = plt.subplots()
    for label, data, color in series:
        mean = np.mean(data)
        stdev = np.std(data)
        weights = np.ones_like(data) / len(data)
        line = ax.hist(data, bins=bins, alpha=0.6, weights=weights, color=color,
                        label=f'{label} (μ={mean:.3f}, σ={stdev:.3f})')
        ax.axvline(mean, color=line[2][0].get_facecolor(), linestyle='--', linewidth=1)

    ax.set_xlabel('Dihedral angle (rad)')
    ax.set_ylabel('Relative Frequency')
    ax.legend()

    plt.xlim(0, 1.4)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f'Saved histogram to {out_path}')


if __name__ == '__main__':
    args = parse_args()

    ab_txt_data = load_txt_dihedral(args.ab_txt)
    cd_txt_data = load_txt_dihedral(args.cd_txt)

    print('Computing LAMMPS pentamer_AB dihedral (AB group)...')
    ab_pkl_dihedral = compute_lammps_group_dihedral(args.ab_pkl, 'AB')

    print('Computing LAMMPS pentamer_CD dihedral (CD group)...')
    cd_pkl_dihedral = compute_lammps_group_dihedral(args.cd_pkl, 'CD')

    plot_histogram(ab_txt_data, cd_txt_data, ab_pkl_dihedral, cd_pkl_dihedral, args.out, args.bins)
