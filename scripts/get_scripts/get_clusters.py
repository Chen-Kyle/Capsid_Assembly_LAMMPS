"""
Builds two csv files based off of the cluster information in the given pkl file


WORK IN PROGRESS

Input arguements:
    path_to_pkl_file                    -- all cluster information

Output
    One csv file of the total clusters
    One csv file of the well-formed clusters
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

def print_all_cluster_data(pkl_data, pkl_file_path):
    """
    Prints out the size of the clusters to the same directory as the pkl file
    for every frame and cluster in that frame

    Uses the format:{Frame#}   {ClusterID}   {Cluster Size}
    """
    print("Printing All Cluster Data...\n")
    out_file_name = pkl_file_path.replace('complete_cluster_data.pkl', 'all_cluster.csv')
    all_cluster_data = pkl_data['all_clusters']

    with open(out_file_name, "w") as f:
        # Header
        f.write("Frame#,ClusterID,Cluster Size\n")

        for frame, cluster_frame_data in all_cluster_data.items():
            for cluster_data in cluster_frame_data:
                cluster_size = len(cluster_data['segids'])/2

                f.write(f"{frame},{cluster_data['cluster_id']},{cluster_size}\n")

    print(f"Saved to: {out_file_name}")
    return

def print_well_formed_cluster_data(pkl_data, pkl_file_path):
    """
    Prints out only the number of correctly bonded dimers to the same directory
    as the pkl file for every frame and cluster in that frame

    Uses the format:{Frame#}   {ClusterID}   {Well Formed Cluster Size}
    """
    print("Printing Well-formed Cluster Data...\n")
    out_file_name = pkl_file_path.replace('complete_cluster_data.pkl', 'well_formed_clusters.csv')
    all_cluster_data = pkl_data['all_well_formed_clusters']

    with open(out_file_name, "w") as f:
        # Header
        f.write("Frame#,ClusterID,Well Formed Cluster Size\n")

        for frame, cluster_frame_data in all_cluster_data.items():
            for cluster_data in cluster_frame_data:
                cluster_size = len(cluster_data['segids'])/2

                f.write(f"{frame},{cluster_data['cluster_id']},{cluster_size}\n")

    print(f"Saved to: {out_file_name}")
    return

if __name__ == '__main__':
    args = parse_args()
    pkl_data = load_pickle(args.pkl)

    print_all_cluster_data(pkl_data, args.pkl)
    print_well_formed_cluster_data(pkl_data, args.pkl)