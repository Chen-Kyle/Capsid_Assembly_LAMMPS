"""
analyze_bonds.py
Runs bond detection on each trajectory and prints bonding probabilities.
"""

import os
import argparse
from detect_bonds import parse_frames, get_probability

HBV_ENM_PATH = os.environ.get("HBV_ENM_PATH", "/home/kyle/2026_Research/HBV_enm")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pdb',        default=f'{HBV_ENM_PATH}/scripts/important_oligomer_pdbs/cg_ABCD_avg.pdb',
                   help='Simulation-start PDB (binded decamer)')
    p.add_argument('--traj_folder',       default=f'{HBV_ENM_PATH}/trajectory_files/ABCD_avg_Enative/Enative_traj_files',
                   help='trajectory folder for dcd files to be analyzed')
    p.add_argument('--contactdir', default=f'{HBV_ENM_PATH}/scripts/contact_files',
                   help='Directory containing A_contacts.txt … D_contacts.txt')
    p.add_argument('--output',     default=f'{HBV_ENM_PATH}/raw_data',
                   help='Output directory for bond analysis data')
    return p.parse_args()

PDB_FILE    = "/home/kyle/2026_Research/HBV_enm/scripts/important_oligomer_pdbs/cg_ABCD_avg.pdb"
CONTACT_DIR = "/home/kyle/2026_Research/HBV_enm/scripts/contact_files"
SCRATCH_DIR = "/home/kyle/2026_Research/HBV_enm/trajectory_files/ABCD_avg_Enative/Enative_traj_files"

CUTOFFS           = [13.5, 14.0]
CONTACTS_PER_BOND = [20, 18, 16]


if __name__ == "__main__":
    args = parse_args()

    with open(args.output, "w") as f:
        print(f"Cutoffs: {CUTOFFS}\nContacts per bonds:{CONTACTS_PER_BOND}\n")
        # Runs for every trajectory file
        for filename in os.listdir(args.traj_folder):
            # Handles naming convention errors
            if "Enative=" not in filename:
                print(f"[skip] {filename}: naming convention incorrect, expected 'Enative=' in filename")
                continue

            Enative = (filename.split("Enative="))[-1].split("_seed")[0]
            traj_name = os.path.join(args.traj_folder, filename, "seg.dcd")

            print(f"{'='*60}")
            print(f"Enative = {Enative}   filename: {traj_name}")

            # Runs for every cutoff value given
            for cutoff in CUTOFFS:
                print(f"\n  cutoff = {cutoff} Å")
                iface_bonds, _ = parse_frames(cutoff, args.pdb, traj_name, args.contactdir)

                for cpb in CONTACTS_PER_BOND:
                    prob = get_probability(iface_bonds, cpb)
                    if not prob:
                        print(f"    contacts_per_bond={cpb:2d}:  no interfaces formed")
                        continue
                    avg_prob  = sum(prob.values()) / len(prob)
                    n_bonded  = sum(1 for p in prob.values() if p > 0)
                    print(f"    contacts_per_bond={cpb:2d}:  "
                            f"mean_prob={avg_prob:.3f}  "
                            f"interfaces_ever_bonded={n_bonded}/{len(prob)}")
                print()
