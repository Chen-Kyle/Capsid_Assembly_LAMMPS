"""
build_fcc_lattice.py
====================
Build an FCC crystal lattice from two PDB subunits.

Subunit AB occupies FCC basis positions 0 and 1  → (0,0,0) and (½,½,0)
Subunit CD occupies FCC basis positions 2 and 3  → (½,0,½) and (0,½,½)

This guarantees exactly N/2 copies of each subunit type per unit cell tile.

Usage
-----
    python build_fcc_lattice.py \\
        --ab subunit_AB.pdb \\
        --cd subunit_CD.pdb \\
        --lattice-param 200.0 \\
        --nx 2 --ny 2 --nz 2 \\
        --output lattice.pdb

Arguments
---------
    --ab             PDB file for the AB-type subunit (chains A+B)
    --cd             PDB file for the CD-type subunit (chains C+D)
    --lattice-param  FCC lattice parameter in Angstroms (edge length of cubic unit cell)
    --nx, --ny, --nz Number of unit cells along each axis (default: 1 each)
    --output         Output PDB filename (default: fcc_lattice.pdb)
    --center         If set, center the entire lattice at the origin before writing
"""

import argparse
import warnings
import os
import random
import math
import numpy as np
import MDAnalysis as mda

warnings.filterwarnings("ignore")

HBV_ENM_PATH = os.environ.get("HBV_ENM_PATH", "/home/kyle/2026_Research/HBV_enm")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
def parse_args():
    p = argparse.ArgumentParser(
        description="Build an FCC lattice from two PDB subunits (AB and CD).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ab",            default=f'{HBV_ENM_PATH}/scripts/important_oligomer_pdbs/cg_A1B1_avg.pdb',
                    help="PDB file for AB subunit")
    p.add_argument("--cd",            default=f'{HBV_ENM_PATH}/scripts/important_oligomer_pdbs/cg_C1D1_avg.pdb',
                   help="PDB file for CD subunit")
    p.add_argument("--lattice-param", default=100,  type=float,
                   help="Cubic lattice spacing parameter in Angstroms (> length of subunit = 85Å)")
    p.add_argument("--box_length", default=500,  type=float,
                   help="Box length in Angstroms (> length of subunit = 85Å)")
    p.add_argument("--N",            default=20,      type=int,
                   help="Number of subunits (should be a multiple of 2 for an equal number of dimers per dimer type)")
    p.add_argument("--output_dir",        default=f"{HBV_ENM_PATH}/scripts/lattice_pdbs",
                   help="Output PDB directory")
    p.add_argument("--center",        action="store_true",
                   help="Center the lattice at the origin before writing")
    return p.parse_args()
 

# ---------------------------------------------------------------------------
# cubic site generation
# ---------------------------------------------------------------------------

def generate_fcc_sites(a: float, n_per_length: int) -> list:
    """
    Return a list of ([x1, y1, z1], [x2, y2, z2]...) for a cubic lattice

    Parameters
    ----------
    a  : lattice parameter (Angstroms)
    N : number of total sites

    Returns
    -------
    List of 3D arrays of shape (3,-) — Cartesian position
    """

    sites = []
    for ix in range(n_per_length):
        for iy in range(n_per_length):
            for iz in range(n_per_length):
                pos = np.array([ix,iy,iz]) * a
                sites.append(pos)
    return sites


# ---------------------------------------------------------------------------
# Per-copy manipulation helpers
# ---------------------------------------------------------------------------

def place_subunit(u: mda.Universe, target: np.ndarray) -> mda.Universe:
    """
    Return a deep copy of universe u with its centre of geometry
    translated to target.
    """
    c = u.copy()
    com = c.atoms.center_of_geometry()
    c.atoms.positions += (target - com)
    return c


def relabel_subunit(u: mda.Universe, chain_map: dict, copy_num: int) -> mda.Universe:
    """
    Return a copy of u with chain IDs remapped via chain_map and segids
    updated to <new_chain_letter><copy_num>.

    chain_map example: {'A': 'A', 'B': 'B'} or {'C': 'C', 'D': 'D'}
    """
    c = u.copy()

    # Remap per-atom chainIDs
    new_chains = np.array([chain_map.get(ch, ch) for ch in c.atoms.chainIDs])
    c.atoms.chainIDs = new_chains

    # Remap segids at the segment level (strip trailing digits, remap letter)
    new_segids = []
    for seg in c.segments.segids:
        letter = seg.rstrip("0123456789")
        new_letter = chain_map.get(letter, letter)
        new_segids.append(f"{new_letter}{copy_num}")
    c.segments.segids = np.array(new_segids)

    return c


# ---------------------------------------------------------------------------
# Main build routine
# ---------------------------------------------------------------------------

def build_lattice(
    ab_pdb: str,
    cd_pdb: str,
    a: float,
    box_length: float,
    n_dimers: int,
    output_dir: str,
    center: bool = False,
) -> None:
    """
    Build the FCC lattice and write it to output_pdb.
    """

    # --- prevents overcrowding ---
    n_per_length = math.ceil(n_dimers**(1/3))
    if (n_per_length*a > box_length):
        raise("Box length is too small for number of dimers. Use fcc lattice script instead")

    # --- load template subunits ---
    print(f"Loading AB subunit: {ab_pdb}")
    u_ab = mda.Universe(ab_pdb)
    print(f"  {len(u_ab.atoms)} atoms | chains: {sorted(set(u_ab.atoms.chainIDs))} "
          f"| segids: {sorted(set(u_ab.atoms.segids))}")

    print(f"Loading CD subunit: {cd_pdb}")
    u_cd = mda.Universe(cd_pdb)
    print(f"  {len(u_cd.atoms)} atoms | chains: {sorted(set(u_cd.atoms.chainIDs))} "
          f"| segids: {sorted(set(u_cd.atoms.segids))}")

    # chain maps — adjust these if your PDB uses different chain letters
    ab_chain_map = {ch: ch for ch in set(u_ab.atoms.chainIDs)}
    cd_chain_map = {ch: ch for ch in set(u_cd.atoms.chainIDs)}

    # --- generate cubic lattice sites ---
    sites = generate_fcc_sites(a, n_per_length)

    print(f"\nLattice: {n_per_length}×{n_per_length}×{n_per_length} unit cells | a = {a} Å")
    print(f"  {n_per_length**3} total sites created ({n_dimers/ 2} AB + {n_dimers/2} CD)")

    # --- place and relabel every copy ---
    all_atom_groups = []
    ab_count = 0
    cd_count = 0
    n_per_type = int(n_dimers/2)

    dimer_order = ['AB'] * n_per_type + ['CD'] * n_per_type
    random.seed(42)
    random.shuffle(dimer_order)

    for i, dimer_type in enumerate(dimer_order):
        pos = sites[i]
        if dimer_type == 'AB':           # AB sublattice
            ab_count += 1
            placed   = place_subunit(u_ab, pos)
            labeled  = relabel_subunit(placed, ab_chain_map, ab_count)
        else:                       # CD sublattice
            cd_count += 1
            placed   = place_subunit(u_cd, pos)
            labeled  = relabel_subunit(placed, cd_chain_map, cd_count)

        all_atom_groups.append(labeled.atoms)

    # --- merge all copies ---
    print(f"\nMerging {ab_count} AB copies + {cd_count} CD copies …")
    merged = mda.Merge(*all_atom_groups)
    total_atoms = len(merged.atoms)
    print(f"  Total atoms in output: {total_atoms}")

    # --- set box dimensions ---
    merged.dimensions = np.array([box_length, box_length, box_length, 90.0, 90.0, 90.0])

    # --- optional: center lattice at origin ---
    if center:
        com = merged.atoms.center_of_geometry()
        merged.atoms.positions -= com
        print(f"  Centered at origin (shifted by {-com})")

    # --- write output ---
    os.makedirs(output_dir, exist_ok=True)
    filename = f'{output_dir}/lattice=cubic_Ndimers={n_dimers}_blength={int(box_length)}.pdb'
    with mda.Writer(filename, total_atoms) as w:
        w.write(merged.atoms)

    print(f"\nOutput written to: {filename}")
    print(f"  Box dimensions:  {box_length:.3f} × {box_length:.3f} × {box_length:.3f} Å")
    print(f"  AB copies (chains {sorted(ab_chain_map.values())}): {ab_count}")
    print(f"  CD copies (chains {sorted(cd_chain_map.values())}): {cd_count}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    args = parse_args()
    build_lattice(
        ab_pdb       = args.ab,
        cd_pdb       = args.cd,
        a            = args.lattice_param,
        box_length   = args.box_length,
        n_dimers     = args.N,
        output_dir   = args.output_dir,
        center       = args.center,
    )