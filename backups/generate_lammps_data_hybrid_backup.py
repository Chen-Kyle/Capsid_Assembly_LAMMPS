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
        --bound    /path/to/decamer_avg.pdb  \
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

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pdb',        default='important_oligomer_pdbs/cg_ABCD_separate.pdb',
                   help='Simulation-start PDB (separated decamer)')
    p.add_argument('--bound',      default='important_oligomer_pdbs/cg_ABCD_avg.pdb',
                   help='Bound-state PDB (used to identify native contact partners)')
    p.add_argument('--conndir',    default='connect_files',
                   help='Directory containing cg_*_connectivity.txt files')
    p.add_argument('--contactdir', default='.',
                   help='Directory containing A_contacts.txt … D_contacts.txt')
    p.add_argument('--Enative',    type=float, default=1.0,
                   help='Native contact energy scale (multiplies all Gaussian Eatt)')
    p.add_argument('--output',     default='decamer.lammps',
                   help='Output LAMMPS data file name')
    p.add_argument('--type', choices=['pairs'], default=None,
                   help='Go model variant. '
                        '"pairs": uses standard pair_style hybrid/overlay table '
                        'with one unique atom type per contact atom — works with '
                        'any LAMMPS installation including Stampede3 modules.')
    return p.parse_args()

# ---------------------------------------------------------------------------
# Harmonic bond builder
# ---------------------------------------------------------------------------

def _find_conn_file(conndir, dname):
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
    dimer_list = []
    for i in range(1, 100):
        if f'A{i}' in segids and f'B{i}' in segids:
            dimer_list.append(f'A{i}B{i}')
        if f'C{i}' in segids and f'D{i}' in segids:
            dimer_list.append(f'C{i}D{i}')
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
    for dname in dimer_list:
        letters = [c for c in dname if c.isalpha()]
        numbers = [c for c in dname if c.isdigit()]
        seg1 = letters[0] + numbers[0]   # e.g. 'A1'
        seg2 = letters[1] + numbers[1]   # e.g. 'B1'
        n_dimer = len(ca.select_atoms(f'segid {seg1} or segid {seg2}'))

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
# Native contact builder  (mirrors setup_system.contact_list_new)
# ---------------------------------------------------------------------------

def _partner_segid_A(monomer_num):
    if (monomer_num - 1) % 5 == 0:
        return f'A{monomer_num + 4}'
    return f'A{monomer_num - 1}'


def _contact_pairs_for_monomer(iface, mn, u_sim, ubound, contactdir):
    """
    Returns list of (i0, j0) 0-based indices into u_sim CA atoms,
    or [] if this interface/monomer combination has no contacts.
    """
    # Identify partner segment from the bound-state structure
    if iface == 'A':
        partner_seg = _partner_segid_A(mn)
        sle = ubound.select_atoms(f'segid {partner_seg}')
    elif iface == 'B':
        sle = ubound.select_atoms(
            f'(around 9 segid B{mn}) and not chainID D and not segid A{mn}')
    elif iface == 'C':
        sle = ubound.select_atoms(
            f'(around 9 segid C{mn}) and not chainID A '
            f'and not chainID B and not segid D{mn}')
    elif iface == 'D':
        sle = ubound.select_atoms(
            f'(around 9 segid D{mn}) and not chainID C and not chainID A')
    else:
        return []

    unique_segs = list(set(sle.segids))
    if not unique_segs:
        return []
    partner_seg = unique_segs[0]
    m1 = f'{iface}{mn}'

    contact_file = os.path.join(contactdir, f'{iface}_contacts.txt')
    if not os.path.exists(contact_file):
        raise FileNotFoundError(f'Contact file not found: {contact_file}')
    clist = pd.read_csv(contact_file, sep='\t', header=None)

    pairs = []
    for _, row in clist.iterrows():
        res1, res2 = int(row[1]), int(row[3])
        g1 = u_sim.select_atoms(f'segid {m1} and resid {res1} and name CA')
        g2 = u_sim.select_atoms(f'segid {partner_seg} and resid {res2} and name CA')
        if len(g1) == 0 or len(g2) == 0:
            return []
        # MDAnalysis ids are 1-based; subtract 1 for 0-based index
        pairs.append((int(g1.ids[0]) - 1, int(g2.ids[0]) - 1))
    return pairs


def build_native_contacts(u_sim, ubound, contactdir):
    """
    Return dict {'A': [(i0,j0), ...], 'B': [...], 'C': [...], 'D': [...]}.
    Monomer numbers are detected from the PDB segIDs, so this works for any
    oligomeric state without needing to iterate up to an arbitrary maximum.
    """
    segids = set(u_sim.select_atoms('name CA').segids)
    monomer_nums = sorted({
        int(''.join(c for c in s if c.isdigit()))
        for s in segids if s and any(c.isdigit() for c in s)
    })

    contacts = {t: [] for t in 'ABCD'}
    for iface in 'ABCD':
        for mn in monomer_nums:
            pairs = _contact_pairs_for_monomer(iface, mn, u_sim, ubound, contactdir)
            contacts[iface].extend(pairs)
    return contacts

# ---------------------------------------------------------------------------
# Gaussian bond potential tables
# ---------------------------------------------------------------------------

TABLE_R_MAX = 300.0   # Å — must exceed the maximum initial inter-dimer separation

def write_gaussian_table(filename, Eatt_kcal, n_points=15000, table_r_max=TABLE_R_MAX):
    """
    Write a LAMMPS table file for the Gaussian native contact.

    E(r) = -Eatt × [A_G exp(-B_G r²) + C_G exp(-D_G r²)]  for r ≤ R_CUT_G (30 Å)
         = 0                                                  for r > R_CUT_G

    table_r_max controls the upper bound of the table:
      - Default (300 Å): required for bond_style table, which must cover the
        maximum initial inter-dimer separation in the separated PDB.
      - 30 Å: correct for pair_style table/pairs (Go model), where the neighbor
        list cutoff handles distance screening and the table only needs to cover
        the physical range where the potential is non-zero.
    """
    r_vals = np.linspace(0.5, table_r_max, n_points)

    with open(filename, 'w') as f:
        f.write(
            f'# Gaussian native contact potential\n'
            f'# E(r) = -Eatt*(A*exp(-B*r^2)+C*exp(-D*r^2)) for r <= {R_CUT_G:.1f} Ang, else 0\n'
            f'# Eatt={Eatt_kcal:.6f}  A={A_GAUSS:.6f}  C={C_GAUSS:.6f}'
            f'  B={B_GAUSS:.6f}  D={D_GAUSS:.6f}\n'
            f'#\n'
            f'GAUSSIAN_NC\n'
            f'N {n_points}\n\n'
        )
        for idx, r in enumerate(r_vals, start=1):
            if r > R_CUT_G:
                E, F = 0.0, 0.0
            else:
                gA = A_GAUSS * np.exp(-B_GAUSS * r**2)
                gC = C_GAUSS * np.exp(-D_GAUSS * r**2)
                E  = -Eatt_kcal * (gA + gC)
                # F = -dE/dr; dE/dr = Eatt * 2r * (B*gA + D*gC)
                F  = -Eatt_kcal * 2.0 * r * (B_GAUSS * gA + D_GAUSS * gC)
            f.write(f'{idx:6d}  {r:12.6f}  {E:16.10f}  {F:16.10f}\n')

# ---------------------------------------------------------------------------
# LAMMPS data file writer
# ---------------------------------------------------------------------------

def write_lammps_data(outfile, positions_ang, harmonic_bonds, native_contacts,
                      contact_atom_types=None):
    """
    Write the LAMMPS data file and return (n_harm_types, iface_to_btype).

    Standard mode (contact_atom_types=None):
      1 atom type. Bond types 1…N_harm (harmonic) + N_harm+1…+4 (Gaussian).

    --type hybrid (contact_atom_types provided):
      1 + N_contact_atoms atom types. Bond types 1…N_harm only. Each atom
      that participates in a native contact gets a unique type so that
      pair_coeff entries can target it exactly without approximation.
    """
    n_atoms = len(positions_ang)
    type_pairs_mode = contact_atom_types is not None

    # Assign a unique harmonic bond type per unique equilibrium distance
    r0_to_type = {}
    harm_typed = []
    for (i0, j0, r0) in harmonic_bonds:
        key = round(r0, 4)
        if key not in r0_to_type:
            r0_to_type[key] = len(r0_to_type) + 1
        harm_typed.append((i0, j0, r0_to_type[key]))
    n_harm_types = len(r0_to_type)

    if type_pairs_mode:
        iface_to_btype = {}
        all_bonds      = list(harm_typed)
        n_bond_types   = n_harm_types
    else:
        iface_to_btype = {iface: n_harm_types + k
                          for k, iface in enumerate('ABCD', start=1)}
        all_bonds = list(harm_typed)
        for iface, pairs in native_contacts.items():
            btype = iface_to_btype[iface]
            for (i0, j0) in pairs:
                all_bonds.append((i0, j0, btype))
        n_bond_types = n_harm_types + 4

    n_bonds = len(all_bonds)

    # Atom types: 1 for standard/go-model; 1+N_contact for type-pairs mode.
    if type_pairs_mode:
        n_atom_types = 1 + len(contact_atom_types)
    else:
        n_atom_types = 1

    # Box: extend coordinates by 10 Å padding on each side
    lo = positions_ang.min(axis=0) - 10.0
    hi = positions_ang.max(axis=0) + 10.0

    with open(outfile, 'w') as f:
        f.write('LAMMPS data file: HBV decamer CG oligomer\n\n')
        f.write(f'{n_atoms} atoms\n')
        f.write(f'{n_bonds} bonds\n\n')
        f.write(f'{n_atom_types} atom types\n')
        f.write(f'{n_bond_types} bond types\n\n')
        f.write(f'{lo[0]:.4f} {hi[0]:.4f} xlo xhi\n')
        f.write(f'{lo[1]:.4f} {hi[1]:.4f} ylo yhi\n')
        f.write(f'{lo[2]:.4f} {hi[2]:.4f} zlo zhi\n\n')

        # All types share the same mass (average amino-acid residue, g/mol).
        # mass * 110.0 in lammps_oligomer.in also works, but listing explicitly
        # here is required when n_atom_types > 1.
        f.write('Masses\n\n')
        for t in range(1, n_atom_types + 1):
            f.write(f'{t} 110.0\n')
        f.write('\n')

        f.write('# columns: atom_id mol_id atom_type x y z\n')
        f.write('Atoms  # bond\n\n')
        for aid, (x, y, z) in enumerate(positions_ang, start=1):
            atype = contact_atom_types.get(aid - 1, 1) if type_pairs_mode else 1
            f.write(f'{aid:6d}  1  {atype}  {x:12.6f}  {y:12.6f}  {z:12.6f}\n')
        f.write('\n')

        f.write('Bonds\n\n')
        for bid, (i0, j0, btype) in enumerate(all_bonds, start=1):
            f.write(f'{bid:8d}  {btype:6d}  {i0+1:6d}  {j0+1:6d}\n')

    return n_harm_types, iface_to_btype

# ---------------------------------------------------------------------------
# Include-file writers
# ---------------------------------------------------------------------------

def write_harmonic_coeffs(filename, r0_to_type, type_pairs=False):
    # bond_style harmonic (used by --type hybrid and default): "bond_coeff N K r0"
    # bond_style hybrid (default mode): "bond_coeff N harmonic K r0"
    with open(filename, 'w') as f:
        f.write('# ENM harmonic bond coefficients\n')
        f.write(f'# K = {K_HARM:.4f} kcal/mol/Ang^2 for all harmonic bonds\n\n')
        for r0_val, btype in sorted(r0_to_type.items(), key=lambda x: x[1]):
            if type_pairs:
                f.write(f'bond_coeff  {btype}  {K_HARM:.4f}  {r0_val:.4f}\n')
            else:
                f.write(f'bond_coeff  {btype}  harmonic  {K_HARM:.4f}  {r0_val:.4f}\n')


def write_native_contact_coeffs(filename, iface_to_btype):
    with open(filename, 'w') as f:
        f.write('# Gaussian native contact bond coefficients\n')
        f.write('# E(r) = -Eatt*(A*exp(-B*r^2)+C*exp(-D*r^2)), tabulated to 30 Ang\n\n')
        for iface, btype in iface_to_btype.items():
            f.write(
                f'bond_coeff  {btype}  table  '
                f'gaussian_native_{iface}.table  GAUSSIAN_NC\n'
            )



def build_contact_atom_types(native_contacts):
    """
    Assign a unique atom type (starting from 2) to every atom that appears
    in any native contact. Type 1 is reserved for non-contact atoms.

    Returns a dict mapping 0-based atom index -> LAMMPS atom type (int >= 2).
    An atom that participates in multiple contacts across different interfaces
    still gets only one type; pair_coeff handles all its partners individually.
    """
    contact_atom_types = {}
    next_type = 2
    for pairs in native_contacts.values():
        for (i0, j0) in pairs:
            if i0 not in contact_atom_types:
                contact_atom_types[i0] = next_type
                next_type += 1
            if j0 not in contact_atom_types:
                contact_atom_types[j0] = next_type
                next_type += 1
    return contact_atom_types


def write_pair_coeffs_by_type(filename, native_contacts, contact_atom_types):
    """
    Write pair_coeff entries for --type hybrid mode.
    One line per native contact pair: pair_coeff T_i T_j table gaussian_native_X.table GAUSSIAN_NC
    T_i and T_j are the unique atom types assigned by build_contact_atom_types().
    No explicit cutoff — LAMMPS uses the table's outer boundary (30 Ang) automatically,
    which is required for compatibility with older LAMMPS versions (e.g. 2018).
    """
    with open(filename, 'w') as f:
        f.write('# Native contact pair coefficients (--type hybrid)\n')
        f.write('# pair_coeff T_i T_j table gaussian_native_X.table GAUSSIAN_NC\n\n')
        for iface, pairs in native_contacts.items():
            for (i0, j0) in pairs:
                ti = contact_atom_types[i0]
                tj = contact_atom_types[j0]
                f.write(f'pair_coeff  {ti:6d}  {tj:6d}  table  '
                        f'gaussian_native_{iface}.table  GAUSSIAN_NC  30.0\n')


def write_forces_lammps(filename, type_pairs=False, n_contacts=0):
    """
    Write forces.lammps — the include file sourced by lammps_oligomer.in.
    Contains all pair_style, pair_coeff, bond_style, and comm_modify settings
    for the chosen mode so that lammps_oligomer.in never needs manual editing.

    Modes:
      default (type_pairs=False):
        Original bond_style table. Requires comm_modify cutoff 305.0 because
        bond tables span the full initial inter-dimer separation (~300 Ang).

      type_pairs=True (--type hybrid):
        Standard pair_style hybrid/overlay table with unique atom types.
        Works on any LAMMPS installation. No comm_modify needed.
    """
    with open(filename, 'w') as f:
        if type_pairs:
            f.write('# Forces: Go model, standard pair_style hybrid/overlay (--type hybrid)\n')
            f.write('# Works with any LAMMPS installation including Stampede3 modules.\n\n')
            if n_contacts > 0:
                f.write('pair_style  hybrid/overlay soft 15.0 table linear 15000\n')
                f.write('pair_coeff  * *  soft  ${A_soft}\n')
                f.write('include     native_contact_pair_coeffs.lammps\n\n')
            else:
                f.write('# No native contacts detected — using plain soft repulsion only.\n')
                f.write('pair_style  soft 15.0\n')
                f.write('pair_coeff  * *  ${A_soft}\n\n')
            f.write('special_bonds lj 1 1 1\n\n')
            f.write('bond_style  harmonic\n')
            f.write('include     harmonic_bond_coeffs.lammps\n')

        else:
            f.write('# Forces: original bond_style table (no --type flag)\n')
            f.write('# comm_modify required: bond tables span 300 Ang for separated start.\n\n')
            f.write('comm_modify cutoff 305.0\n\n')
            f.write('pair_style  soft 15.0\n')
            f.write('pair_coeff  * *  ${A_soft}\n\n')
            f.write('special_bonds lj 1 1 1\n\n')
            f.write('bond_style  hybrid harmonic table linear 15000\n')
            f.write('include     harmonic_bond_coeffs.lammps\n')
            f.write('include     native_contact_coeffs.lammps\n')

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f'Reading simulation PDB: {args.pdb}')
    u_sim   = mda.Universe(args.pdb)
    ca      = u_sim.select_atoms('name CA')
    positions = ca.positions.copy()   # Å
    print(f'  {len(positions)} CA atoms')

    print(f'Reading bound-state PDB: {args.bound}')
    ubound = mda.Universe(args.bound)

    # Auto-detect dimer list from PDB segIDs (works for any oligomeric state)
    dimer_list = detect_dimer_list(u_sim)
    if not dimer_list:
        raise ValueError(
            f'No AB or CD dimer pairs found in {args.pdb}. '
            'Check that segIDs follow the A1/B1/C1/D1 naming convention.')
    print(f'  Detected {len(dimer_list)} dimers: {dimer_list}')

    print('Building harmonic ENM bonds...')
    harmonic_bonds = build_harmonic_bonds(u_sim, dimer_list, args.conndir)
    print(f'  {len(harmonic_bonds)} bonds')

    print('Building native contacts...')
    native_contacts = build_native_contacts(u_sim, ubound, args.contactdir)
    for iface, pairs in native_contacts.items():
        print(f'  Interface {iface}: {len(pairs)} contact pairs')

    type_pairs = (args.type == 'pairs')
    table_rmax = 30.0 if type_pairs else 300.0

    print('Writing Gaussian potential tables...')
    for iface, scale in INTERFACE_SCALE.items():
        Eatt = scale * args.Enative
        fname = f'gaussian_native_{iface}.table'
        write_gaussian_table(fname, Eatt, table_r_max=table_rmax)
        print(f'  {fname}  (Eatt = {Eatt:.4f} kcal/mol, scale = {scale}×{args.Enative})')
    if type_pairs:
        print(f'  (Go model: tables span 0.5–30 Ang only)')

    # For --type hybrid, assign a unique atom type to every contact atom.
    contact_atom_types = build_contact_atom_types(native_contacts) if type_pairs else None

    print(f'Writing LAMMPS data file: {args.output}')
    n_harm_types, iface_to_btype = write_lammps_data(
        args.output, positions, harmonic_bonds, native_contacts,
        contact_atom_types=contact_atom_types)
    if type_pairs:
        n_pairs = sum(len(v) for v in native_contacts.values())
        n_ctypes = len(contact_atom_types)
        print(f'  {n_harm_types} harmonic bond types, {n_pairs} native contacts as pairs')
        print(f'  {1 + n_ctypes} atom types ({n_ctypes} contact atoms + 1 default)')
    else:
        print(f'  {n_harm_types} harmonic bond types, 4 Gaussian bond types')
        print(f'  {len(harmonic_bonds) + sum(len(v) for v in native_contacts.values())} total bonds')

    print('Writing harmonic_bond_coeffs.lammps...')
    r0_to_type = {}
    for (i0, j0, r0) in harmonic_bonds:
        key = round(r0, 4)
        if key not in r0_to_type:
            r0_to_type[key] = len(r0_to_type) + 1
    write_harmonic_coeffs('harmonic_bond_coeffs.lammps', r0_to_type, type_pairs=type_pairs)

    if type_pairs:
        print('Writing native_contact_pair_coeffs.lammps...')
        write_pair_coeffs_by_type('native_contact_pair_coeffs.lammps',
                                  native_contacts, contact_atom_types)
    else:
        print('Writing native_contact_coeffs.lammps...')
        write_native_contact_coeffs('native_contact_coeffs.lammps', iface_to_btype)

    print('Writing forces.lammps...')
    n_contacts = sum(len(v) for v in native_contacts.values())
    write_forces_lammps('forces.lammps', type_pairs=type_pairs, n_contacts=n_contacts)

    print('\nDone. To run:')
    print(f'  lmp -in lammps_oligomer.in -var Erepulsion 1.0 -var myseed 42 -var nsteps 1000000')


if __name__ == '__main__':
    main()
