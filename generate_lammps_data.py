"""
generate_lammps_data.py
Generates all LAMMPS input files needed by lammps_oligomer.in.

Reads the same source files as run_oligomer_simulation.py and writes:
    decamer.lammps                   -- atom positions + bond topology
    gaussian_native_{A,B,C,D}.table  -- tabulated Gaussian pair potentials
    harmonic_bond_coeffs.lammps      -- bond_coeff lines for ENM harmonic bonds
    native_contact_coeffs.lammps     -- bond_coeff lines for native contacts

Usage:
    python generate_lammps_data.py \
        --pdb      /path/to/decamer_sep.pdb  \
        --conndir  /path/to/connect_files    \
        --contactdir /path/to/contacts       \
        --Enative  1.0
"""

import argparse
import os
import numpy as np
import pandas as pd
import MDAnalysis as mda

# ---------------------------------------------------------------------------
# Unit conversions and physical constants
# ---------------------------------------------------------------------------
KJ_TO_KCAL = 0.239006          # 1 kJ/mol = 0.239006 kcal/mol
NM_TO_ANG  = 10.0              # 1 nm = 10 Å

# Harmonic bond spring constant (kcal/mol/Å²)
# OpenMM uses E = (k/2)(r-r0)² with k = 41840 kJ/(mol·nm²)
# LAMMPS uses   E = K  (r-r0)²  so K = k/2
K_HARM = (41840.0 / 2.0) * KJ_TO_KCAL / (NM_TO_ANG**2)  # = 50.0 kcal/mol/Å²

# Gaussian native contact shape parameters (converted to Å, kcal/mol)
A_GAUSS = 4.6   * KJ_TO_KCAL         # 1.09943 kcal/mol
C_GAUSS = 8.368 * KJ_TO_KCAL         # 1.99983 kcal/mol
B_GAUSS = 10.0  / NM_TO_ANG**2       # 0.10 Å⁻²  (10.0 nm⁻²)
D_GAUSS = 1.0   / NM_TO_ANG**2       # 0.01 Å⁻²  (1.0 nm⁻²)
R_CUT_G = 3.0   * NM_TO_ANG          # 30.0 Å    (3.0 nm)

# Eatt prefactor per interface type (multiplied by Enative)
INTERFACE_SCALE = {'A': 1.17, 'B': 1.11, 'C': 1.30, 'D': 1.00}

# Right-column chain type for each contacts file (derived from spatial query exclusions)
#   A_contacts.txt : A chain contacts A chain
#   B_contacts.txt : B chain contacts C chain  (D excluded in original query)
#   C_contacts.txt : C chain contacts D chain  (A and B excluded, not-D{mn} leaves D other)
#   D_contacts.txt : D chain contacts B chain  (C and A excluded)
CONTACT_PARTNER_CHAIN = {'A': 'A', 'B': 'C', 'C': 'D', 'D': 'B'}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pdb',        default='decamer_sep.pdb',
                   help='Simulation-start PDB (separated decamer)')
    p.add_argument('--conndir',    default='connect_files',
                   help='Directory containing cg_*_connectivity.txt files')
    p.add_argument('--contactdir', default='contact_files',
                   help='Directory containing A_contacts.txt … D_contacts.txt')
    p.add_argument('--Enative',    type=float, default=1.0,
                   help='Native contact energy scale (multiplies all Gaussian Eatt)')
    p.add_argument('--output_dir',     default='lammps_out',
                   help='Output LAMMPS data file name')
    return p.parse_args()

# ---------------------------------------------------------------------------
# Harmonic bond builder
# ---------------------------------------------------------------------------

def _find_conn_file(conndir, dname):
    """
    Looks at the name of a dimer like 'A1B1' and then returns the correct path
    to its respective connectivity file
    """
    letters = [c for c in dname if c.isalpha()]
    mtype = ''.join(letters)           # 'AB' or 'CD'
    candidates = [
        os.path.join(conndir, f'cg_{dname}_connectivity.txt'),
        os.path.join(conndir, f'cg_{mtype}_avg_connectivity.txt'),
        os.path.join(conndir, f'cg_{mtype}_connectivity.txt'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f'No connectivity file for dimer {dname} in {conndir}. '
        f'Tried: {candidates}')


def detect_dimer_list(u_sim):
    """
    Auto-detect which AB and CD dimers are present in the PDB from its segIDs.
    Returns a list like ['A1B1', 'C1D1', 'A2B2', ...] ordered by monomer number,
    interleaved AB/CD as in the original decamer setup.
    """
    segids = set(u_sim.select_atoms('name CA').segids)
    n_segs = len(u_sim.select_atoms('name CA'))
    n_dimers = int(n_segs/596) + 2
    dimer_list = []
    for i in range(1, n_dimers):
        if f'A{i}' in segids and f'B{i}' in segids:
            dimer_list.append(f'A{i}B{i}')
        if f'C{i}' in segids and f'D{i}' in segids:
            dimer_list.append(f'C{i}D{i}')
    print(dimer_list)
    return dimer_list


def build_harmonic_bonds(u_sim, dimer_list, conndir):
    """
    Return list of (i0, j0, r0_ang) tuples (0-based atom indices).
    i0/j0 are indices into u_sim.select_atoms('name CA').
    The per-dimer offset is counted from the actual segid atom counts so the
    function works for any oligomeric state (trimer, pentamer, decamer, …).
    """
    ca = u_sim.select_atoms('name CA')
    pos = ca.positions   # Å, shape (N, 3)

    bonds = []
    offset = 0
    i0 = 0
    j0 = 0
    loop_runs = 0
    dname = ""
    for dname in dimer_list:
        letters = [c for c in dname if c.isalpha()]
        numbers = [c for c in dname if c.isdigit()]
        seg1 = letters[0] + numbers[0]   # e.g. 'A1'
        seg2 = letters[1] + numbers[1]   # e.g. 'B1'
        n_dimer = 298 #len(ca.select_atoms(f'segid {seg1} or segid {seg2}'))

        conn_file = _find_conn_file(conndir, dname)
        # columns: local_atom1 local_atom2 [flag] (1-indexed within dimer)
        raw = np.loadtxt(conn_file, dtype=int)
        if raw.ndim == 1:
            raw = raw[np.newaxis, :]
        for row in raw:
            i0 = offset + row[0] - 1
            j0 = offset + row[1] - 1
            r0 = float(np.linalg.norm(pos[j0] - pos[i0]))
            bonds.append((i0, j0, r0))

        offset += n_dimer
    return bonds

# ---------------------------------------------------------------------------
# Native contact builder
# ---------------------------------------------------------------------------

def build_atom_type_map(u_sim):
    """
    Assigns a unique LAMMPS atom type to each (chain_letter, resnum) combination.
    chain_letter is the first character of the segid (e.g. 'A' from 'A1').

    Returns:
      type_map -- dict: (chain_letter, resnum) -> [list of 0-based atom indices]
                  All atoms sharing a (chain, resnum) key get the same LAMMPS type.
                  LAMMPS type number = enumeration position in the dict (1-based).
    """
    ca = u_sim.select_atoms('name CA')
    type_map = {}
    for atom in ca:
        key = (atom.segid[0], atom.resid)
        if key not in type_map:
            type_map[key] = []
        type_map[key].append(int(atom.id) - 1)
    return type_map


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
# Gaussian bond potential tables
# ---------------------------------------------------------------------------

TABLE_R_MAX = 32.0    # Å — slightly above R_CUT_G (30 Å); pair_style uses this as neighbor cutoff

def write_gaussian_table(filename, Eatt_kcal, n_points=20000):
    """
    Writes a LAMMPS bond_style table file for the Gaussian native contact.

    E(r) = -Eatt × [A_G exp(-B_G r²) + C_G exp(-D_G r²)]  for r ≤ R_CUT_G (30 Å)
         = 0                                                  for r > R_CUT_G

    The table spans 0.5 – TABLE_R_MAX Å so that bond_style table never throws
    "Bond length > table outer cutoff" even for atoms that start far apart in
    the separated initial structure.  With 15000 points the spacing is ~0.02 Å,
    giving ~1450 non-zero points in the physical 0–30 Å range.
    """
    r_vals = np.linspace(0.001, TABLE_R_MAX, n_points)
    r_cut = 8 # Angstroms -  from Smiriti's thesis
    Eatt_kcal = 0
    with open(filename, 'w') as f:
        f.write(
            f'# Gaussian native contact potential\n'
            f'# E(r) = -Eatt*(A*exp(-B*r^2)+C*exp(-D*r^2)) for r <= {R_CUT_G:.1f} Ang, else 0\n'
            f'# Eatt={Eatt_kcal:.6f}  A={A_GAUSS:.6f}  C={C_GAUSS:.6f}'
            f'  B={B_GAUSS:.6f}  D={D_GAUSS:.6f}  r_cut={r_cut}\n'
            f'#\n'
            f'GAUSSIAN_NC\n'
            f'N {n_points}\n\n'
        )
        for idx, r in enumerate(r_vals, start=1):
            if r >= R_CUT_G:
                E, F = 0.0, 0.0
            else:
                dr = r - r_cut
                gA = A_GAUSS * np.exp(-B_GAUSS * dr**2)
                gC = C_GAUSS * np.exp(-D_GAUSS * dr**2)
                E  = -Eatt_kcal * (gA + gC)
                # F = -dE/dr; dE/dr = Eatt * 2*(r-r_cut) * (B*gA + D*gC)
                F  = -Eatt_kcal * 2.0 * dr * (B_GAUSS * gA + D_GAUSS * gC)
            f.write(f'{idx:6d}  {r:12.6f}  {E:16.10f}  {F:16.10f}\n')

# ---------------------------------------------------------------------------
# LAMMPS data file writer
# ---------------------------------------------------------------------------

def write_lammps_data(outfile, positions_ang, harmonic_bonds, type_map, atom_to_molid):
    """
    Writes decamer.lammps the LAMMPS data file.

    LAMMPS type number for each (chain_letter, resnum) key is its 1-based
    position in type_map (insertion order, guaranteed in Python 3.7+).
    Atoms sharing a key get the same type — monomer-agnostic by design.

    atom_to_molid: {0-based atom index -> mol_id} so that neigh_modify
    exclude molecule/intra all suppresses intra-chain A-A self-interactions.

    Only harmonic ENM bonds are written. Native contacts are pair interactions
    written separately by write_native_contact_pair_coeffs.

    Returns r0_to_type: {r0_val: bond_type_int} for write_harmonic_coeffs.
    """
    # Build atom_index -> lammps_type from type_map enumeration order
    key_to_ltype = {key: i for i, key in enumerate(type_map, start=1)}
    atom_to_ltype = {}
    for key, atom_ids in type_map.items():
        ltype = key_to_ltype[key]
        for aid in atom_ids:
            atom_to_ltype[aid] = ltype

    n_atoms      = len(positions_ang)
    n_atom_types = len(type_map)

    r0_to_type = {}
    harm_typed = []
    for (i0, j0, r0) in harmonic_bonds:
        key = round(r0, 4)
        if key not in r0_to_type:
            r0_to_type[key] = len(r0_to_type) + 1
        harm_typed.append((i0, j0, r0_to_type[key]))
    n_bond_types = len(r0_to_type)
    n_bonds      = len(harm_typed)

    lo = positions_ang.min(axis=0) - 50.0
    hi = positions_ang.max(axis=0) + 50.0

    with open(outfile, 'w') as f:
        f.write('LAMMPS data file: HBV capsid CG Go model\n\n')
        f.write(f'{n_atoms} atoms\n')
        f.write(f'{n_bonds} bonds\n\n')
        f.write(f'{n_atom_types} atom types\n')
        f.write(f'{n_bond_types} bond types\n\n')
        f.write(f'{lo[0]:.4f} {hi[0]:.4f} xlo xhi\n')
        f.write(f'{lo[1]:.4f} {hi[1]:.4f} ylo yhi\n')
        f.write(f'{lo[2]:.4f} {hi[2]:.4f} zlo zhi\n\n')

        f.write('Masses\n\n')
        for t in range(1, n_atom_types + 1):
            f.write(f'{t}  110.0\n')
        f.write('\n')

        f.write('Atoms  # bond  (atom_id mol_id atom_type x y z)\n\n')
        for aid, (x, y, z) in enumerate(positions_ang, start=1):
            atype  = atom_to_ltype[aid - 1]
            molid  = atom_to_molid[aid - 1]
            f.write(f'{aid:6d}  {molid}  {atype:4d}  {x:12.6f}  {y:12.6f}  {z:12.6f}\n')
        f.write('\n')

        f.write('Bonds\n\n')
        for bid, (i0, j0, btype) in enumerate(harm_typed, start=1):
            f.write(f'{bid:8d}  {btype:6d}  {i0+1:6d}  {j0+1:6d}\n')

    return r0_to_type

# ---------------------------------------------------------------------------
# Include-file writers
# ---------------------------------------------------------------------------

def write_harmonic_coeffs(filename, r0_to_type):
    """
    Writes harmonic_bond_coeff which stores bond information for LAMMPS to initialize from
    """

    with open(filename, 'w') as f:
        f.write('# ENM harmonic bond coefficients\n')
        f.write('# bond_coeff N  harmonic  K(kcal/mol/Ang^2)  r0(Ang)\n')
        f.write(f'# K = {K_HARM:.4f} kcal/mol/Ang^2 for all harmonic bonds\n\n')
        for r0_val, btype in sorted(r0_to_type.items(), key=lambda x: x[1]):
            f.write(f'bond_coeff  {btype}  {K_HARM:.4f}  {r0_val:.4f}\n')


def write_native_contact_pair_coeffs(output_dir, contacts, type_map):
    """
    Writes pair_coeff entries for all native contact type pairs.
    type_map keys are (chain_letter, resnum); LAMMPS type = 1-based position in dict.
    Any atom whose key maps to type1 attracts any atom whose key maps to type2,
    regardless of which specific monomer they belong to.
    """
    key_to_ltype = {key: i for i, key in enumerate(type_map, start=1)}

    outfile = f'{output_dir}/native_contact_pair_coeffs.lammps'
    print(f'Writing {outfile}...')

    with open(outfile, 'w') as f:
        
        f.write('# NB: This file does not reflect the change made which negates ex: A1 A1 interactions\n')
        f.write('# Gaussian native contact pair coefficients\n')
        f.write('# pair_coeff type1 type2 table <file> GAUSSIAN_NC\n\n')
        seen = set()
        for iface, pairs in contacts.items():
            fname = f'{output_dir}/gaussian_native_{iface}.table'
            for (chain1, res1, chain2, res2) in pairs:
                t1 = key_to_ltype.get((chain1, res1))
                t2 = key_to_ltype.get((chain2, res2))
                if t1 is None or t2 is None:
                    continue
                key = (min(t1, t2), max(t1, t2))
                if key in seen:
                    continue
                seen.add(key)
                f.write(f'pair_coeff  {t1}  {t2}  table  {fname}  GAUSSIAN_NC\n')

# ---------------------------------------------------------------------------
# PDB connectivity writer
# ---------------------------------------------------------------------------

def write_connectivity_to_pdb(conn_data, pdb_file, cutoff_distance):
    '''
    Write out bond connectivity to a new pdb file along with CA positions.
    For now do this manually, eventually we should use MDAnalysis for more complicated situations
    '''
    with open(pdb_file, 'r') as f:
        old_pdb_lines = f.readlines()

    if not any(l.startswith('CONECT') for l in old_pdb_lines):
        new_pdb_file = pdb_file.replace('.pdb', f'_with_connectivity_cutoff={cutoff_distance}.pdb')
        with open(new_pdb_file, 'w') as f:
            for line in old_pdb_lines[:-1]:
                f.write(line)
            for i in range(conn_data.shape[0]):
                # This logic will fail when the number of atoms exceeds 1m since there will no 
                # longer be enough space to write out the full serial id of the atom
                f.write(f'CONECT{int(conn_data[i][0]):5d}{int(conn_data[i][1]):5d}\n')
            f.write(old_pdb_lines[-1])
        print(f'  Written: {new_pdb_file}')
    else:
        print('PDB file already has CONECT information. Not adding any.')

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Reading simulation PDB: {args.pdb}')
    u_sim     = mda.Universe(args.pdb)
    ca        = u_sim.select_atoms('name CA')
    positions = ca.positions.copy()
    print(f'  {len(positions)} CA atoms')

    dimer_list = detect_dimer_list(u_sim)
    if not dimer_list:
        raise ValueError(
            f'No AB or CD dimer pairs found in {args.pdb}. '
            'Check that segIDs follow the A1/B1/C1/D1 naming convention.')
    print(f'  Detected {len(dimer_list)} dimers: {dimer_list}')

    print('Building harmonic ENM bonds...')
    harmonic_bonds = build_harmonic_bonds(u_sim, dimer_list, args.conndir)
    print(f'  {len(harmonic_bonds)} bonds')

    print('Writing PDB with bond connectivity...')
    conn_data = np.array([[i0 + 1, j0 + 1] for (i0, j0, _) in harmonic_bonds])
    write_connectivity_to_pdb(conn_data, args.pdb, 'from_connectivity_files')

    print('Building native contacts...')
    native_contacts = build_native_contacts(args.contactdir)
    print("  Loaded pattern possibilities (expected 20 per interface)")
    for iface, pairs in native_contacts.items():
        print(f'  Interface {iface}: {len(pairs)} contact patterns')

    print('Building atom type map...')
    type_map = build_atom_type_map(u_sim)
    print(f'  {len(type_map)} unique (chain, resnum) atom types')

    segid_to_molid = {s: i for i, s in enumerate(sorted(set(ca.segids)), start=1)}
    atom_to_molid  = {int(atom.id) - 1: segid_to_molid[atom.segid] for atom in ca}
    print(f'  {len(segid_to_molid)} chains → mol IDs: {segid_to_molid}')

    print('Writing Gaussian potential tables...')
    for iface, scale in INTERFACE_SCALE.items():
        Eatt = scale * args.Enative
        fname = f'{args.output_dir}/gaussian_native_{iface}.table'
        write_gaussian_table(fname, Eatt)
        print(f'  {fname}  (Eatt = {Eatt:.4f} kcal/mol)')

    print(f'Writing LAMMPS data file: {args.output_dir}/decamer.lammps')
    r0_to_type = write_lammps_data(
        f'{args.output_dir}/decamer.lammps', positions, harmonic_bonds, type_map, atom_to_molid)
    print(f'  {len(r0_to_type)} harmonic bond types')
    print(f'  {len(type_map)} atom types (one per chain-letter + resnum combination)')

    print(f'Writing {args.output_dir}/harmonic_bond_coeffs.lammps...')
    write_harmonic_coeffs(f'{args.output_dir}/harmonic_bond_coeffs.lammps', r0_to_type)

    write_native_contact_pair_coeffs(args.output_dir, native_contacts, type_map)

    print('\nDone.')


if __name__ == '__main__':
    main()