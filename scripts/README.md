# Unbiased MD simulations for HBV elastic network model using LAMMPS (ported over from Smiriti Pradhan)
- `run_oligomer_simulation.py` is the python script for running the dynamics.
- In addition, need the **starting pdb** file, files specifying the elastic network **bond connectivity**, reference pdb file that defines the **native contacts**.
- Force field for coarse-grained simulation from paper `     Manish Gupta et al., Critical mechanistic features of HIV-1 viral capsid assembly.Sci. Adv.9,eadd7434(2023) .DOI:10.1126/sciadv.add7434`

- Make sure to add HBV_ENM_PATH to your .bashrc giving the path to the project folder ie: /home/your_name/project_name/
