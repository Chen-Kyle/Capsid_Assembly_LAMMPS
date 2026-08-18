# Unbiased MD simulations for HBV elastic network model using LAMMPS
(ported over from Smriti Pradhan)

- `generate_lammps_data.py` is the python script for writing the lammps file that you need to run the simulation
- `lammps_oligomer.in` is the lammps script for running the actual simulation where it will reference decamer.lammps for its other parameters
- In addition, need the **starting pdb** file, files specifying the elastic network **bond connectivity**, reference pdb file that defines the **native contacts**.
- Force field for coarse-grained simulation from paper `     Manish Gupta et al., Critical mechanistic features of HIV-1 viral capsid assembly.Sci. Adv.9,eadd7434(2023) .DOI:10.1126/sciadv.add7434`
- Make sure to add HBV_ENM_PATH to your .bashrc giving the path to the project folder ie: /home/your_name/project_directory/

All scripts read their defaults off of `$HBV_ENM_PATH` (falling back to
`/home/kyle/2026_Research/HBV_enm` if the env var isn't set), so as long as
that's set correctly every script below can be run with no flags at all
using the checked-in example files.

---

## The general workflow

```
 1. Build a starting structure
    build_cubic_lattice_pdb.py / build_fcc_lattice_pdb.py   (only if you need a
    fresh lattice of dimers; otherwise use one of the pdbs already in
    important_oligomer_pdbs/)
             |
             v
 2. Run the simulation
    generate_lammps_data.py  -->  decamer.lammps, gaussian_native_*.table,
                                   harmonic_bond_coeffs.lammps,
                                   native_contact_pair_coeffs.lammps
             |
             v
    lmp -in lammps_oligomer.in  -->  seg.dcd  (the trajectory)
    (run_lammps.sh / local_run_lammps.sh / batch_run_lammps.sh wrap both of
    the above steps)
             |
             v
 3. Analyze the trajectory into one pkl
    full_traj_analysis.py --pdb ... --traj seg.dcd --contactdir ...
        -->  complete_cluster_data.pkl   (written next to seg.dcd)
    (run_detect_clusters.sh / run_detect_clusters_batch.sh /
    local_run_detect_clusters_batch.sh wrap this step, usually queued
    automatically at the end of run_lammps.sh)
             |
             v
 4. Extract the numbers you actually want, per pkl, into small CSVs
    get_scripts/get_binding_angles.py  --pkl complete_cluster_data.pkl --> binding_angles.csv
    get_scripts/get_spike_angles.py    --pkl complete_cluster_data.pkl --> spike_angles.csv
    get_scripts/get_dihedrals.py       --pkl complete_cluster_data.pkl --> dihedral_angles.csv
    get_scripts/get_clusters.py        --pkl complete_cluster_data.pkl --> all_cluster.csv, well_formed_clusters.csv
    (all four are written into the same directory as the pkl)
             |
             v
 5. Plot the CSVs
    plt_scripts/plot_csv_histogram.py     --csv dihedral_angles.csv / binding_angles.csv / spike_angles.csv
    plt_scripts/plot_csv_clustersize.py   --csv all_cluster.csv / well_formed_clusters.csv
```

So for a single trajectory, end to end, it looks like:

```bash
python generate_lammps_data.py --pdb important_oligomer_pdbs/abcd_capsid.pdb --Enative 1.0 --output_dir lammps_out
lmp -in lammps_oligomer.in -var output_dir lammps_out -var nsteps 10000000 -var myseed 42

python full_traj_analysis.py --pdb important_oligomer_pdbs/abcd_capsid.pdb --traj lammps_out/seg.dcd --contacts 20

python get_scripts/get_binding_angles.py --pkl lammps_out/complete_cluster_data.pkl
python get_scripts/get_spike_angles.py   --pkl lammps_out/complete_cluster_data.pkl
python get_scripts/get_dihedrals.py      --pkl lammps_out/complete_cluster_data.pkl
python get_scripts/get_clusters.py       --pkl lammps_out/complete_cluster_data.pkl

python plt_scripts/plot_csv_histogram.py   --csv lammps_out/dihedral_angles.csv
python plt_scripts/plot_csv_clustersize.py --csv lammps_out/all_cluster.csv --type all
```

`local_run_lammps.sh` chains steps 2-3 for you locally, and `run_lammps.sh`
does the same on SLURM, plus automatically queues `run_detect_clusters.sh`
for step 3 once the LAMMPS run finishes.

### Scripts that do NOT follow this workflow

A few scripts predate the pkl-based pipeline above, or were written for a
one-off analysis, and either read the raw pdb/trajectory directly (instead
of `--pkl`) or read from hardcoded, non-standard paths. They still work, but
don't expect them to take a `--pkl complete_cluster_data.pkl` argument the
way the rest of the pipeline does:

- **`get_scripts/get_distances.py`** — takes `--pdb`/`--traj` directly, not `--pkl`, and re-parses the raw `contact_files/*_contacts.txt` itself instead of reusing `full_traj_analysis.py`'s output. Also currently **broken**: the last line of `__main__` calls `plot_distance_data(args.dnames, outfile, args.out)` with 3 arguments, but the function only accepts 2 (`dname, h5_path`) — this raises `TypeError` every time it's run to completion. Its two output HDF5 files also have hardcoded filenames (`h5_files/_all_distances.h5`, `h5_files/_native_contacts_distances.h5`) that aren't tagged by `--dnames`, so re-running it for a different dimer pair silently overwrites the previous pair's output.
- **`analyze_clusters.py`** and **`analyze_last_frame_clusters.py`** — these do read cluster pkls, but are standalone plotting/diagnostic tools rather than part of the get_scripts -> CSV -> plt_scripts chain (see their own sections below for details, including `analyze_last_frame_clusters.py` deliberately reimplementing `full_traj_analysis.py`'s logic rather than consuming its pkl).
- **`plt_scripts/plot_apo_vs_lammps_dihedral_histogram.py`**, **`plot_cluster_panels.py`**, **`plot_conc_sweep_v4_heatmap.py`** — these plot directly from raw `cluster_data.pkl`/`complete_cluster_data.pkl` files (globbed across a whole sweep directory) rather than from the small per-run CSVs in `lammps_out/`, and have hardcoded, one-off paths baked in (specific job IDs, specific sweep names, external `/home/kyle/storage/...` paths). See their sections below.

---

## 1. Building a starting structure

### `build_cubic_lattice_pdb.py`
Places randomly-ordered copies of an AB dimer and a CD dimer PDB onto a
simple cubic lattice inside a fixed box, at a given AB:CD percentage, and
writes the merged structure as one PDB. Useful for seeding a concentration
sweep (see `batch_run_lammps.sh` below).

```bash
python build_cubic_lattice_pdb.py \
    --ab important_oligomer_pdbs/cg_A1B1_avg.pdb \
    --cd important_oligomer_pdbs/cg_C1D1_avg.pdb \
    --box_length 1000 --N 60 --ratio 50 \
    --output_dir lattice_pdbs --center
```
- `--ab` (default `important_oligomer_pdbs/cg_A1B1_avg.pdb`), `--cd` (default `important_oligomer_pdbs/cg_C1D1_avg.pdb`) — the two subunit PDBs to tile
- `--box_length` (float, default `1000`) — cubic box side length, Å
- `--N` (int, default `60`) — total number of subunits to place
- `--ratio` (int, default `50`) — percent of subunits that are AB (rest are CD)
- `--output_dir` (default `lattice_pdbs`), `--center` (flag) — center the lattice on the origin

Writes `{output_dir}/lattice=cubic_Ndimers={N}_blength={box_length}.pdb`.

*Note:* the module's own top-of-file docstring is copy-pasted from the FCC
script below (wrong script name, wrong flag names) — go by the `--help`
output / the flags listed here, not the docstring. There's also a bug at
line ~149: `raise("Box length is too small...")` raises a bare string
instead of an exception class, so that box-size guard doesn't actually work.

### `build_fcc_lattice_pdb.py`
Same idea, but tiles a proper face-centered-cubic lattice (4 sites per unit
cell — always 2 AB + 2 CD, i.e. a fixed 1:1 ratio) instead of a random cubic
packing.

```bash
python build_fcc_lattice_pdb.py \
    --ab important_oligomer_pdbs/cg_A1B1_avg.pdb \
    --cd important_oligomer_pdbs/cg_C1D1_avg.pdb \
    --lattice_param 100 --nl 5 \
    --output_dir lattice_pdbs --center
```
- `--ab`, `--cd` — same defaults as above
- `--lattice_param` (float, default `100`) — unit cell size, Å
- `--nl` (int, default `5`) — cells per axis (1→4 sites, 2→32, 3→108, 4→256, 5→500)
- `--output_dir` (default `lattice_pdbs`), `--center` (flag)

Writes `{output_dir}/lattice=fcc_Ndimers={n_sites}_blength={box_x}.pdb`.

*Note:* same doc-drift issue as the cubic script — the docstring's example
usage (`--lattice-param`, `--output`) doesn't match the real flags
(`--lattice_param`, `--output_dir`).

Both scripts write into `important_oligomer_pdbs/../lattice_pdbs`, which is
exactly the directory `batch_run_lammps.sh` sweeps over — run one of these
first if you want to build a fresh set of starting configurations for a
concentration sweep.

---

## 2. Running the simulation

### `generate_lammps_data.py`
Converts a coarse-grained starting PDB plus connectivity/contact tables into
the LAMMPS input files needed by `lammps_oligomer.in`.

```bash
python generate_lammps_data.py \
    --pdb important_oligomer_pdbs/pentamer_avg.pdb \
    --conndir connect_files \
    --contactdir contact_files \
    --Enative 1.0 \
    --output_dir lammps_out
```
- `--pdb` (default `important_oligomer_pdbs/pentamer_avg.pdb`) — starting PDB, segids must follow the `A1/B1/C1/D1...` convention
- `--conndir` (default `connect_files`) — directory with `cg_*_connectivity.txt` (elastic-network bond topology)
- `--contactdir` (default `contact_files`) — directory with `{A,B,C,D}_contacts.txt` (native contacts)
- `--Enative` (float, default `1.0`) — scales the native-contact attraction strength
- `--output_dir` (default `lammps_out`) — where the generated files go
- `--buildconn` (flag) — also write a copy of the input PDB with `CONECT` records added

Writes, into `--output_dir`: `decamer.lammps` (atoms + bonds),
`gaussian_native_{A,B,C,D}.table` (tabulated Gaussian pair potentials),
`harmonic_bond_coeffs.lammps`, `native_contact_pair_coeffs.lammps`.

*Notes:* the docstring/header comments in this file and in `run_lammps.sh`
mention a `native_contact_coeffs.lammps` output and a `--type` flag
(pairs/hybrid mode) that don't exist in the current script — leftover from
an earlier version. Also, `n_dimer` (atoms per dimer, used when building
native-contact bonds) is hardcoded to `298` with the real MDAnalysis-based
computation commented out, so this will silently be wrong if you ever use a
dimer topology with a different atom count.

### `lammps_oligomer.in`
The actual LAMMPS input script: reads the files generated above, sets up the
hybrid pair-style/bond-style force field, minimizes energy, then runs
Langevin dynamics and dumps the trajectory.

```bash
lmp -in lammps_oligomer.in \
    -var output_dir lammps_out \
    -var nsteps 10000000 \
    -var myseed 42 \
    -var Erepulsion 1.0
```
Variables (all overridable via `-var`, defaults shown): `output_dir=""`
(must be supplied — `read_data ${output_dir}/decamer.lammps` will fail
otherwise), `nsteps=10000000`, `myseed=42`, `Erepulsion=1.0`.

Reads `${output_dir}/decamer.lammps`, `${output_dir}/native_contact_pair_coeffs.lammps`,
`${output_dir}/harmonic_bond_coeffs.lammps`. Writes `minimized.lammps`,
`minimized.lammpstrj`, `seg_final.lammps`, `seg.restart` (all in the working
directory), and the trajectory `${output_dir}/seg.dcd` (dumped every 5000
steps).

*Known limitation (flagged in-file):* there's no equilibration MD run
between energy minimization and the start of the production dump — the
minimize step only removes bad contacts/clashes, it doesn't bring the system
to thermal equilibrium at 300 K before `seg.dcd` starts recording. If the
original OpenMM workflow equilibrated first, that step is currently missing
here.

### `run_lammps.sh` (SLURM)
Runs one full simulation — `generate_lammps_data.py` then `lmp`, then queues
the cluster-detection step — for one seed/pdb/Enative/nsteps combination.

```bash
sbatch run_lammps.sh <seed> <pdb_path> <Enative> <nsteps> <output_dir_toplevel> <use_job_id>
# e.g.
sbatch run_lammps.sh 42 important_oligomer_pdbs/abcd_capsid.pdb 1.0 100000000
```
Positional args (all optional, in order): `seed` (default `42`), `pdb`
(default `important_oligomer_pdbs/abcd_capsid.pdb`), `Enative` (default
`1.0`), `nsteps` (default `100000000`), `output_dir_toplevel` (default
`${SCRATCH}/HBV_enm/Enative=${Enative}_seed=${seed}`), `use_job_id` (default
`yes` — nests output under `${SLURM_JOB_ID}/` when set).

Runs `python generate_lammps_data.py ...`, then
`mpirun -n $SLURM_NTASKS lmp -in lammps_oligomer.in ...`, appends a line to
`master.log`, and at the end automatically submits
`sbatch run_detect_clusters.sh ${PDB} ${output_dir}/seg.dcd` to kick off step
3 of the pipeline. Called in a loop by `batch_run_lammps.sh`.

*Note:* the header comment lists outputs (`native_contact_coeffs.lammps`,
`native_contacts.pairs`, a `--type pairs/hybrid` mode) that don't correspond
to anything `generate_lammps_data.py` actually produces — stale comment
carried over from an older version of the generator.

### `local_run_lammps.sh`
Non-SLURM equivalent of `run_lammps.sh` for running on a single machine —
also chains in step 3 (`full_traj_analysis.py`) automatically at the end.
No arguments; everything is hardcoded at the top of the file
(`PDB=important_oligomer_pdbs/abcd_capsid.pdb`, `output_dir=lammps_out`,
`Enative=0.5`, `nsteps=10000000`) — edit the script directly to change them.

```bash
bash local_run_lammps.sh
```

*Note:* its header comment is copy-pasted from `run_lammps.sh` and still
references `${SLURM_JOB_ID}`/`${SCRATCH}`, which don't apply to this local
script.

### `batch_run_lammps.sh`
Sweep driver: submits one `run_lammps.sh` SLURM job per
`(seed, pdb, Enative)` combination, sweeping every PDB file found in
`lattice_pdbs/` (built via the `build_*_lattice_pdb.py` scripts above)
across a fixed list of Enative values.

No arguments; everything is hardcoded at the top: `seed_vals=(42)`,
`PDB_dir=lattice_pdbs`, `Enative_vals=(0.1 0.2 ... 1.5)` (15 values),
`nsteps=1000000`, `output_tag=Enative_concentration_sweep`.

```bash
bash batch_run_lammps.sh
```

Submits nothing itself but produces one `sbatch run_lammps.sh ...` call per
combination, so make sure `lattice_pdbs/` is populated first.

---

## 3. Turning a trajectory into `complete_cluster_data.pkl`

### `full_traj_analysis.py`
The core analysis script. Walks every frame of a trajectory, determines
which native-contact bonds are formed (using a Gaussian-cutoff neighbor
list, rebuilt every 10 frames), builds connected-component clusters of
bonded dimers, computes per-interface binding/spike angles, and also builds
a stricter "well-formed" cluster set (same as the raw clusters, but an edge
only counts if its binding angle is in `(0.7, 1.5)` rad and its spike angle
is in `(0, 1.2)` rad). Persistent cluster IDs are then assigned across
frames by segid-overlap.

```bash
python full_traj_analysis.py \
    --pdb important_oligomer_pdbs/cg_ABCD_avg.pdb \
    --traj lammps_out/seg.dcd \
    --contactdir claude_computed_contact_files \
    --contacts 19 \
    --output_dir ""
```
- `--pdb` (default `important_oligomer_pdbs/cg_ABCD_avg.pdb`) — simulation-start PDB
- `--traj` (default `lammps_out/seg.dcd`) — trajectory to analyze
- `--contactdir` (default `claude_computed_contact_files`) — directory with `{A,B,C,D}_contacts_with_computed.txt` (per-residue-pair cutoff distances, a superset of the plain `contact_files/*_contacts.txt` used by `generate_lammps_data.py`)
- `--contacts` (int, default `19`) — minimum number of native-contact bonds between two segids to count that interface as "active"
- `--output_dir` (default `''`) — prefix prepended to the traj file's own directory when deciding where to write the pkl

**Output:** `complete_cluster_data.pkl`, written next to the trajectory file
(the path is built by taking `traj_file` and replacing the literal string
`"seg.dcd"` with `"complete_cluster_data.pkl"` — so this only works cleanly
if your trajectory file is actually named `seg.dcd`). Contains a dict with
keys `iface_bonds`, `iface_res_data`, `all_clusters`,
`all_well_formed_clusters`, `interface_data`, `pdb_file`, `traj_file` — this
is the one file every script in section 4 reads from.

### `run_detect_clusters.sh` (SLURM)
Runs `full_traj_analysis.py` once, on a single trajectory.

```bash
sbatch run_detect_clusters.sh <pdb_path> <traj_file>
```
`$1` defaults to `important_oligomer_pdbs/abcd_capsid.pdb`, `$2` defaults to
a specific old scratch path — always pass both explicitly. Calls
`python full_traj_analysis.py --pdb ... --traj ... --contacts 20`. This is
what `run_lammps.sh` submits automatically once a simulation finishes.

*Note:* the header comment says this produces `cluster_data.pkl` and
several of the echo statements still say "detect_clusters.py" — both are
leftover names from an earlier version; the script that actually runs, and
the file it actually writes, are `full_traj_analysis.py` and
`complete_cluster_data.pkl`.

### `run_detect_clusters_batch.sh` (SLURM) / `local_run_detect_clusters_batch.sh`
Batch versions: recursively `find`s every `seg.dcd` under a sweep directory
and runs `full_traj_analysis.py` on each one in sequence (not a true SLURM
array — one job/process walks the whole list). For each trajectory, the
matching starting PDB is looked up by job ID in `master.log`
(`JOBID:<id> ... PDB File:<path>`, as written by `run_lammps.sh`/
`batch_run_lammps.sh`), falling back to a default PDB if no match is found.

```bash
# SLURM
sbatch run_detect_clusters_batch.sh [PDB_fallback] [sweep_dir] [master_log]
# local
bash local_run_detect_clusters_batch.sh [PDB_fallback] [sweep_dir] [master_log]
```
Defaults differ only in `sweep_dir` (a `/scratch0/...` path for the SLURM
version vs. a local `/home/kyle/storage/...` path for the local version) and
in the SLURM version having an `#SBATCH` header. Otherwise the two scripts
are line-for-line identical. Each writes `complete_cluster_data.pkl` next to
every `seg.dcd` it finds.

---

## 4. Extracting CSVs from the pkl (`get_scripts/`)

All four of these take a single `--pkl` flag (default
`lammps_out/complete_cluster_data.pkl`) and write their output CSV into the
**same directory as the pkl** (by substituting the `complete_cluster_data.pkl`
suffix — so the pkl must be named exactly that for the output path
substitution to work).

### `get_scripts/get_binding_angles.py`
```bash
python get_scripts/get_binding_angles.py --pkl lammps_out/complete_cluster_data.pkl
```
Reads `interface_data` from the pkl, writes `binding_angles.csv`:
`Frame#,Interface,Binding Angle` (radians, 6 decimals), one row per
interface per frame.

### `get_scripts/get_spike_angles.py`
```bash
python get_scripts/get_spike_angles.py --pkl lammps_out/complete_cluster_data.pkl
```
Same shape as above but for spike angle. Writes `spike_angles.csv`:
`Frame#,Interface,Spike Angle`.

### `get_scripts/get_dihedrals.py`
```bash
python get_scripts/get_dihedrals.py --pkl lammps_out/complete_cluster_data.pkl
```
Reads `all_well_formed_clusters` from the pkl to determine which dimers are
validly formed each frame, then opens the pdb/traj referenced *inside* the
pkl (`pkl_data['pdb_file']`/`pkl_data['traj_file']`, falling back to a
sibling `seg.dcd` if that path doesn't resolve) to actually compute each
dimer's dihedral angle. Writes `dihedral_angles.csv`:
`Frame#,Dimer,Dihedral Angle`.

### `get_scripts/get_clusters.py`
```bash
python get_scripts/get_clusters.py --pkl lammps_out/complete_cluster_data.pkl
```
Reads `all_clusters` and `all_well_formed_clusters`, writes two CSVs:
`all_cluster.csv` (`Frame#,ClusterID,Cluster Size`) and
`well_formed_clusters.csv` (`Frame#,ClusterID,Well Formed Cluster Size`).
Cluster size = number of dimers (`len(segids)/2`, since each dimer
contributes 2 segids). *Marked "WORK IN PROGRESS" in its own docstring* —
it runs end-to-end, but treat the output format as subject to change.

### Scripts in `get_scripts/` that deviate from this pattern
See "Scripts that do NOT follow this workflow" above for
`get_distances.py` and `get_dihedrals_copy.py` — both take a raw
`--pdb`/`--traj` instead of `--pkl`.

---

## 5. Plotting the CSVs (`plt_scripts/`)

### `plt_scripts/plot_csv_histogram.py`
General-purpose histogram plotter for any of the 3-column
`Frame#,<label>,<value>` CSVs above (`dihedral_angles.csv`,
`binding_angles.csv`, `spike_angles.csv`).
```bash
python plt_scripts/plot_csv_histogram.py --csv lammps_out/dihedral_angles.csv --output_dir raw_data
```
- `--csv` (default `lammps_out/binding_angles.csv`)
- `--output_dir` (default `raw_data`)

Axis labels/title/output filename are all auto-derived from the CSV's own
header row. Saves `{output_dir}/{value_column_name}_histogram.png` (dpi 150)
and also opens it interactively via `plt.show()`.

### `plt_scripts/plot_csv_clustersize.py`
Plots cluster size vs. time from `get_clusters.py`'s output, two ways: max
cluster size per frame, and one line per persistent `ClusterID`.
```bash
python plt_scripts/plot_csv_clustersize.py --type wellformed --output_dir raw_data
```
- `--csv` (default resolved from `--type`: `lammps_out/all_cluster.csv` or `lammps_out/well_formed_clusters.csv`)
- `--type` (`all` or `wellformed`, default `all`)
- `--output_dir` (default `raw_data`)

Saves `{output_dir}/{type}_max_cluster_size_vs_time.png` and
`{output_dir}/{type}_cluster_size_by_id_vs_time.png` (dpi 150).

### One-off / sweep-specific plotting scripts
These three are not part of the standard per-run CSV pipeline — they read
raw pkl files (often globbed across a whole parameter sweep) and have
hardcoded paths tied to specific past runs. Treat them as templates to copy
and edit for a new sweep, not as general reusable tools:

- **`plot_apo_vs_lammps_dihedral_histogram.py`** — overlays experimental "apo" dihedral histograms (`--ab_txt`/`--cd_txt`, from `data_from_carolina/`) against dihedral angles recomputed from two specific hardcoded `cluster_data.pkl` runs (`--ab_pkl`/`--cd_pkl`, defaulting to specific job IDs under `/home/kyle/storage/.../dihedral_trajs/`). Saves one PNG to `--out`.
- **`plot_cluster_panels.py`** — meant to build a grid of cluster-size-vs-time subplots across an Enative x blength sweep, reading `master.log` and globbing `raw_data/cluster_data/**/*.pkl`. Its `parse_args()` is dead code (never called from `__main__`, and has a stray invalid `p.add_argument('')`) — the script actually runs entirely off hardcoded module-level paths pointing at an old `Enative_concentration_sweep_v2` sweep. Needs repair before pointing it at a new sweep.
- **`plot_conc_sweep_v4_heatmap.py`** — builds a blength-vs-Enative heatmap of final-frame max cluster size across a sweep directory (`--pkl_dir`, `--master_log`, `--cluster_key` = `all_well_formed_clusters` or `all_clusters`, `--output_dir`, `--output_name`). More parameterized than the other two, but still has a hardcoded default sweep path and a hardcoded `if enative == 2.0: skip` exclusion baked into the loop.

---

## Standalone cluster-inspection tools

These two also read cluster pkls, but sit outside the CSV pipeline (no CSV
output) — useful for interactive/one-off inspection rather than batch
processing.

### `analyze_clusters.py`
```bash
python analyze_clusters.py --file <path/to/complete_cluster_data.pkl> --output_dir <dir> --interactive
```
- `--file` — path to the pkl (no default; effectively required)
- `--output_dir` (default `../raw_data/cluster_data`)
- `--interactive` (flag) — without this flag, the script just loads the pkl, prints its keys, and exits; **the actual plotting functions (interface-contact time series, per-interface distance histograms, cluster-size-vs-time) are only reachable through the interactive y/n menu.**

Saves PNGs (`iface_contacts.png`, `dists_<interface>.png`,
`clusters_vs_time.png`, `clusters_vs_time_comparison.png`) to
`--output_dir`, and also opens each interactively via `plt.show()`.

### `analyze_last_frame_clusters.py`
A fast, single-frame variant of `full_traj_analysis.py` — computes native
contacts, binding/spike angles, and well-formed clusters for only the
**last** frame of a trajectory, instead of walking the whole thing.
Deliberately self-contained (per its own docstring) rather than built on top
of `full_traj_analysis.py`, so it duplicates that script's contact-loading,
angle, and clustering logic rather than reading a pkl.

```bash
python analyze_last_frame_clusters.py \
    --pdb important_oligomer_pdbs/abcd_capsid.pdb --traj lammps_out/seg.dcd \
    --contactdir claude_computed_contact_files --contacts 20 \
    --binding_range 0.0,1.5 --spike_range 0.0,1.0
```
Prints a summary (`n_clusters`, `max_cluster_size_dimers`, etc.) to stdout —
writes no file. `--traj` is required (there is no default).

---

## Supporting data directories

- `important_oligomer_pdbs/` — starting structures (single dimers, pentamers, the full ABCD decamer, whole-capsid pdbs) used as `--pdb` throughout.
- `contact_files/` — plain `{A,B,C,D}_contacts.txt` native-contact tables, consumed by `generate_lammps_data.py` and `get_distances.py`.
- `claude_computed_contact_files/` — the same contacts plus a computed per-pair cutoff distance column, consumed by `full_traj_analysis.py`/`analyze_last_frame_clusters.py`.
- `connect_files/` — `cg_*_connectivity.txt` elastic-network bond topology, consumed by `generate_lammps_data.py`.
- `lattice_pdbs/` — output of `build_cubic_lattice_pdb.py`/`build_fcc_lattice_pdb.py`, input to `batch_run_lammps.sh`.
- `lammps_out/` — the default location for everything a single run produces: the generated LAMMPS input files, `seg.dcd`, `complete_cluster_data.pkl`, and the CSVs from `get_scripts/`.

A few other paths referenced by scripts above (`master.log`, `h5_files/`,
`dihedral_lammps_out/`) don't exist in a fresh checkout — they're created on
demand the first time a script that needs them is run.

`backups/` (old copies of `generate_lammps_data.py` and
`lammps_oligomer.in`) and a few now-superseded top-level scripts
(`detect_clusters.py`, `get_binding_angles.py`, `get_dihedrals.py`,
`get_distances.py`, `get_spike_angle.py`) have been removed — their
replacements are `full_traj_analysis.py` and the scripts under
`get_scripts/` documented above.
