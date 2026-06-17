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

Files that get overwritten:
    - Gaussian tables

"""

import argparse
import os
import numpy as np
import pandas as pd
import MDAnalysis as mda

job_id = os.environ.get("SLURM_JOB_ID")
if job_id != None:
    os.makedirs(job_id, exist_ok=True)
    job_id = f"{job_id}/"
    print("SLURM_JOB_ID found")
else:
    job_id = ""
    print("No SLURM_JOB_ID found")
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
    p.add_argument('--enm_cutoff', type=float, default=7.5,
                   help='Distance cutoff (Å) for ENM harmonic bonds between CA atoms')
    p.add_argument('--contactdir', default='.',
                   help='Directory containing A_contacts.txt … D_contacts.txt')
    p.add_argument('--spring_constant', default=10,
                    help='spring constant in kcal/mol/Angstrom^2')
    p.add_argument('--Enative',    type=float, default=1.0,
                   help='Native contact energy scale (multiplies all Gaussian Eatt)')
    p.add_argument('--output',     default='decamer.lammps',
                   help='Output LAMMPS data file name')
    return p.parse_args()

# ---------------------------------------------------------------------------
# PDB connectivity writer
# ---------------------------------------------------------------------------

def write_connectivity_to_pdb(conn_data, pdb_file, cutoff_distance):
    '''
    Write out bond connectivity to a new pdb file along with CA positions.
    For now do this manually, eventually we should use MDAnalysis for more complicated situations
    '''
    
    with open(pdb_file,'r') as f:
        old_pdb_lines = f.readlines()
        
    if not(any(l.startswith('CONECT') for l in old_pdb_lines)):
    
        new_pdb_file = pdb_file.replace('.pdb','_with_connectivity_cutoff={cutoff_distance}.pdb')
        with open(new_pdb_file,'w') as f:
            for line in old_pdb_lines[:-1]:
                f.write(line)
            for i in range(conn_data.shape[0]):
                f.write(f'CONECT{int(conn_data[i][0]):5d}{int(conn_data[i][1]):5d}\n')# % (conn_data[i][0], conn_data[i][1]))
            f.write(old_pdb_lines[-1])
    else:
        print('PDB file already has CONECT information. Not adding any.')

# ---------------------------------------------------------------------------
# Harmonic bond builder
# ---------------------------------------------------------------------------

def extract_positions_and_atominfo(pdb_file):
    with open(pdb_file) as f:
        lines = f.readlines()
    atoms = [line for line in lines if line.startswith('ATOM')]
    xpos = []
    ypos = []
    zpos = []
    chain = []
    segname = []
    for l in atoms:
        # Fixes the string issue from lack of spacing
        xpos.append(float(l[30:38]))
        ypos.append(float(l[38:46]))
        zpos.append(float(l[46:54]))
        chain.append(l[22])
        segname.append(l[73:74])
    return np.c_[xpos, ypos, zpos], np.c_[chain, segname]

def extract_atom_ids(pdb_file):
    with open(pdb_file) as f:
        lines = f.readlines()
    atoms = [line for line in lines if line.startswith('ATOM')]
    ids = []
    for l in atoms:
        pieces = l.split()
        ids.append(int(pieces[1]))
    return ids

def get_connectivity(pdb_file, cutoff):
    
    ind1_list = []
    ind2_list = []
    #compute distances between CA within distance in pdb
    pos, atominfo = extract_positions_and_atominfo(pdb_file)
    for i in range(pos.shape[0]-1):
        for j in range(i+1, pos.shape[0]):
            #filters out false bonds by checking chains against segnames
            if ((atominfo[i][0] == 'A' and atominfo[j][0] == 'C') or
                (atominfo[i][0] == 'C' and atominfo[j][0] == 'A') or
                (atominfo[i][0] == 'B' and atominfo[j][0] == 'D') or
                (atominfo[i][0] == 'D' and atominfo[j][0] == 'B') or
                (atominfo[i][1] == atominfo[j][1])):

                r0 = float(np.linalg.norm(pos[j] - pos[i]))
                if r0<=float(cutoff):
                    ind1_list.append(i+1)
                    ind2_list.append(j+1)
                
    #Return a numpy array of connected atom indices
    return np.c_[ind1_list,ind2_list]

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

def write_gaussian_table(filename, Eatt_kcal, n_points=15000, table_r_max=30):
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
    r_vals = np.linspace(0.25, table_r_max, n_points)

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
    print(f"Saved to: {filename}")

# ---------------------------------------------------------------------------
# LAMMPS data file writer
# ---------------------------------------------------------------------------

def write_lammps_data(outfile, positions_ang, connectivity,
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
    n_bonds = connectivity.shape[0]

    iface_to_btype = {} #UH OH I think we will need to rebuild this functionality in write native contacts 

    n_atom_types = 1 + len(contact_atom_types)
    n_bond_types = connectivity.shape[0]

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
            atype = contact_atom_types.get(aid - 1, 1)
            f.write(f'{aid:6d}  1  {atype}  {x:12.6f}  {y:12.6f}  {z:12.6f}\n')
        f.write('\n')

        #Write out bond information
        f.write('Bonds # (bond_id bond_type atom_id_1 atom_id_2)\n')
        f.write('\n')
        for i in range(connectivity.shape[0]):
            f.write('%d %d %d %d\n' % (i+1, i+1, connectivity[i][0], connectivity[i][1]))
        f.write('\n')
    print(f"Saved to: {outfile}")

# ---------------------------------------------------------------------------
# Include-file writers
# ---------------------------------------------------------------------------

def write_lammps_harmonic_coeffs(myfile, connectivity, params, recompute_connectivity=False):
    
    #If no connectivity provided, then compute it via a 
    #distance-based criterion.
    if recompute_connectivity==True:
        connectivity = get_connectivity(params.pdb, params.enm_dist_cutoff)

    ids = extract_atom_ids(params.pdb)
    pos, atominfo = extract_positions_and_atominfo(params.pdb)
    r0_list = []
    for i in range(connectivity.shape[0]):
        i1 = int(connectivity[i][0]-1)
        i2 = int(connectivity[i][1]-1)
        r0 = np.linalg.norm(pos[i2][:]-pos[i1][:])
        #print(i, i1+1, i2+1, r0)
        r0_list.append(r0)
    with open(myfile,'w') as f:
        f.write('#ENM harmonic bond coefficients\n')
        f.write('#format: bond_coeff ${bond_type} ${K}(kcal/mol/Ang^2) ${r0}(Ang)\n')
        f.write('\n')
        for i in range(connectivity.shape[0]):
            f.write('bond_coeff %d %.04f %.04f\n' % (i+1, float(params.spring_constant), r0_list[i]))
    print(f"Saved to: {myfile}")

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
    print(f"Saved to: {filename}")

def write_forces_lammps(filename, n_contacts=0):
    """
    Write forces.lammps — the include file sourced by lammps_oligomer.in.
    Contains all pair_style, pair_coeff, and comm_modify settings
    for the chosen mode so that lammps_oligomer.in never needs manual editing.
    """
    with open(filename, 'w') as f:
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
    print(f"Saved to: {filename}")

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

    print('Building native contacts...')
    native_contacts = build_native_contacts(u_sim, ubound, args.contactdir)
    for iface, pairs in native_contacts.items():
        print(f'  Interface {iface}: {len(pairs)} contact pairs')

    table_rmax = 30.0
    print('Writing Gaussian potential tables...')
    for iface, scale in INTERFACE_SCALE.items():
        Eatt = scale * args.Enative
        fname = f'{job_id}gaussian_native_{iface}.table'
        write_gaussian_table(fname, Eatt, table_r_max=table_rmax)
        print(f'  {fname}  (Eatt = {Eatt:.4f} kcal/mol, scale = {scale}×{args.Enative})')

    contact_atom_types = build_contact_atom_types(native_contacts)

    #Add connectivity information to pdb file if not there
    connectivity = get_connectivity(args.pdb, args.enm_cutoff)
    write_connectivity_to_pdb(connectivity, args.pdb, args.enm_cutoff)

    print('Writing harmonic_bond_coeffs.lammps...')
    write_lammps_harmonic_coeffs(f'{job_id}harmonic_bond_coeffs.lammps', connectivity, args)

    print('Writing native_contact_pair_coeffs.lammps...')
    write_pair_coeffs_by_type(f'{job_id}native_contact_pair_coeffs.lammps',
                                native_contacts, contact_atom_types)
    
    print('Writing forces.lammps...')
    n_contacts = sum(len(v) for v in native_contacts.values())
    write_forces_lammps(f'{job_id}forces.lammps', n_contacts=n_contacts)

    # WRITES LAMMPS FILE: decamer.lammps
    output_file = f"{job_id}{args.output}"
    print(f'Writing LAMMPS data file: {output_file}')
    n_harm_types = write_lammps_data(
        output_file, positions, connectivity, contact_atom_types)
    n_pairs = sum(len(v) for v in native_contacts.values())
    n_ctypes = len(contact_atom_types)
    print(f'  {n_harm_types} harmonic bond types, {n_pairs} native contacts as pairs')
    print(f'  {1 + n_ctypes} atom types ({n_ctypes} contact atoms + 1 default)')

    print('\nDone. To run:')
    print(f'  lmp -in lammps_oligomer.in -var Erepulsion 1.0 -var myseed 42 -var nsteps 1000000')


if __name__ == '__main__':
    main()