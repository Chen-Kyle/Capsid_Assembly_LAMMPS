"""
Input arguements:
    path_to_pkl_file                    -- all cluster information

Output
    A txt file containing the binding angles which displays the data like so:
    Frame#, Interface, Binding Angle
"""

import argparse
import os
import MDAnalysis as mda
import numpy as np
import pickle

import warnings
warnings.filterwarnings("ignore")

HBV_ENM_PATH = os.environ.get("HBV_ENM_PATH", "/home/kyle/2026_Research/HBV_enm")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pkl',
                   help='Path to the pickle file', default=f'{HBV_ENM_PATH}/scripts/lammps_out/complete_cluster_data.pkl')
    return p.parse_args()


def load_pickle(path):
    """
    Loads the data from the pickle file
    """
    with open(path, 'rb') as f:
        return pickle.load(f)

def print_binding_angle_data(pkl_data, pkl_file_path):
    """
    Prints out the binding angle data to the same directory as the pkl file
    Uses the format:{Frame#}   {Interface}   {Binding_angle}
    """
    print("Printing Binding Angles...\n")
    out_file_name = pkl_file_path.replace('complete_cluster_data.pkl', 'binding_angles.csv')
    interface_data = pkl_data['interface_data']
    with open(out_file_name, "w") as f:
        # Header
        f.write("Frame#,Interface,Binding Angle\n")

        for frame in interface_data.keys():
            for interface in interface_data[frame].keys():
                value = interface_data[frame][interface]['binding_angle']
                f.write(f"{frame},{interface},{value:.6f}\n")

    print(f"Saved to: {out_file_name}")
    return

if __name__ == '__main__':
    args = parse_args()
    pkl_data = load_pickle(args.pkl)

    print_binding_angle_data(pkl_data, args.pkl)