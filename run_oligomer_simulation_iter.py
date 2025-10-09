from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout, exit, stderr
import numpy as np
import argparse
import time
import numpy as np
import random
import setup_system
import MDAnalysis as mda
import os
home_dire=os.environ["WEST_SIM_ROOT"]
forcefield=ForceField('charmm36.xml','charmm36/water.xml')
class ForceReporter(object):
    def __init__(self, file, reportInterval):
        self._out = open(file, 'w')
        self._reportInterval = reportInterval

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, False, False, True, False, None)

    def report(self, simulation, state):
        forces = state.getForces().value_in_unit(kilojoules/mole/nanometer)
        for f in forces:
            self._out.write('%g %g %g\n' % (f[0], f[1], f[2]))

class ParameterReader:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Parameter Reader')
        #self.parser.add_argument('--psfpath', type=str, help='path for psf file') 
        #self.parser.add_argument('--pdbpath', type=str, help='path for pdb file')
        self.parser.add_argument('--Erepulsion', type=float, help='strength of soft-cosine repulsion in kJ/mol ')
        self.parser.add_argument('--Enative', type=float, help='enegry deptth for morse potential of native contacts')
        self.parser.add_argument('--run', type=int, help='run number')
        # Add more parameters as needned
        #self.parser.add_argument('--nrun', type=int, help='run number')
       
    def read_parameters(self):
        args = self.parser.parse_args()
        return args



class MDsteps():
    def __init__(self,pdbname):
        self.pdb=PDBFile(pdbname)
    def new_system(self,padding_dist):
        #add periodic box with intended padding
        coords = self.pdb.positions
        min_crds = [coords[0][0], coords[0][1], coords[0][2]]
        max_crds = [coords[0][0], coords[0][1], coords[0][2]]
        for coord in coords:
            min_crds[0] = min(min_crds[0], coord[0])
            min_crds[1] = min(min_crds[1], coord[1])
            min_crds[2] = min(min_crds[2], coord[2])
            max_crds[0] = max(max_crds[0], coord[0])
            max_crds[1] = max(max_crds[1], coord[1])
            max_crds[2] = max(max_crds[2], coord[2])
        system=forcefield.createSystem(self.pdb.topology,nonbondedMethod=CutoffNonPeriodic)
        system.setDefaultPeriodicBoxVectors(Vec3(18*nanometer,0,0),Vec3(0,18*nanometer,0),Vec3(0,0,18*nanometer))
        return system
    def create_harmonic_bonds(self,system,bond_strength,dimer_list):
        #define bond strength and equilibrium bond distance
        harmonic_bond=HarmonicBondForce()
        #num_bonds=harmonic_bond_og.getNumBonds()
        #Read bonded parameters and change the bond strength
        for i in range(len(dimer_list)):
            dimer_bonds_txt=np.loadtxt(home_dire+'/cg_'+dimer_list[i]+'_avg_connectivity.txt').T
            dimer_bond_number=len(dimer_bonds_txt[0][:])
            for bond_number in range(dimer_bond_number):
                index1=298*i+int(dimer_bonds_txt[0][bond_number])-1
                index2=298*i+int(dimer_bonds_txt[1][bond_number])-1
                pos1=self.pdb.positions[index1]/nanometer
                pos2=self.pdb.positions[index2]/nanometer
                delta=pos2-pos1
                distance = np.sqrt(np.dot(delta, delta))
                r0=distance
                kappa_bond =bond_strength*kilojoule/(nanometer**2*mole)
                harmonic_bond.addBond(index1,index2,r0,kappa_bond)
        #harmonic_bond.setUsesPeriodicBoundaryConditions(True)
#print(f"harmonic bond force group: {harmonic_bond.getForceGroup()}")
#        print(f"number of h bonds={harmonic_bond.getNumBonds()}")
        system.addForce(harmonic_bond)
                #return system
    def remove_nonbonded_forces(self,system):
        #remove all predefined non-bonded forces if we don't want them
        nonbonded_og=system.getForce(4)
        for index in range(nonbonded_og.getNumParticles()):
            charge,sigma,epsilon=nonbonded_og.getParticleParameters(index)
            charge=0
            epsilon=0
            sigma=0
            nonbonded_og.setParticleParameters(index,charge,sigma,epsilon)
    def add_cosine_repulsion(self,system,energy_repulsion):
        r_cutoff=1.5*nanometer
        energy_repulsion_c=0.1184*kilojoule/(mole)
        energy_expr_repulsion="energy_scale*(1+cos((PI*r)/r_nc))"
        force_repulsion = openmm.CustomNonbondedForce(energy_expr_repulsion)
        force_repulsion.addGlobalParameter("energy_scale", energy_repulsion*energy_repulsion_c)
        force_repulsion.addGlobalParameter("PI", np.pi)
        force_repulsion.addGlobalParameter("r_nc",r_cutoff)
        force_repulsion.setNonbondedMethod(openmm.NonbondedForce.CutoffNonPeriodic)
        force_repulsion.setCutoffDistance(r_cutoff)
        num_particles=system.getNumParticles()
        for i in range(num_particles):
            force_repulsion.addParticle()
        system.addForce(force_repulsion)
    
    def add_gaussian_nativec(self,system,energy_attraction,u):
        r_cutoff_g=5.0*nanometer
        expr_gaussian_native_contacts1="-Eatt1*(A*exp(-B*r^2)+C*exp(-D*r^2))*step(r_ncg-r)"
        force_native_contacts1=openmm.CustomNonbondedForce(expr_gaussian_native_contacts1)
        force_native_contacts1.addGlobalParameter("Eatt1",energy_attraction)
        force_native_contacts1.addGlobalParameter("A",4.6*kilojoule/mole)
        force_native_contacts1.addGlobalParameter("B",10.0/(nanometer**2))
        force_native_contacts1.addGlobalParameter("C",8.368*kilojoule/mole)
        force_native_contacts1.addGlobalParameter("D",1.0/(nanometer**2))
        force_native_contacts1.addGlobalParameter("r_ncg",r_cutoff_g)
    
        force_native_contacts1.setNonbondedMethod(openmm.NonbondedForce.CutoffNonPeriodic)
        force_native_contacts1.setCutoffDistance(r_cutoff_g)
        #Native contact pairs according to all-atom simulations
        ABCDpairs=setup_system.native_contact_list('A2','A1',u)
        ABCDpairs=ABCDpairs+setup_system.native_contact_list('B1','C1',u)
        ABCDpairs=ABCDpairs+setup_system.native_contact_list('D1','B2',u)
        ABCDpairs=ABCDpairs+setup_system.native_contact_list('C1','D10',u)
        ABCDpairs=ABCDpairs+setup_system.native_contact_list('C10','D23',u)
        ABCDpairs=ABCDpairs+setup_system.native_contact_list('C23','D1',u)
        print(ABCDpairs)
        print(len(ABCDpairs))
        ABCDpairs=np.asarray(ABCDpairs)-1
        number_native_contacts=len(ABCDpairs)
        for i in range(number_native_contacts):
            d1index=ABCDpairs[i][0]
            d2index=ABCDpairs[i][1]
            force_native_contacts1.addInteractionGroup([d1index],[d2index])
        num_particles=system.getNumParticles()
        for i in range(num_particles):
            force_native_contacts1.addParticle()
        system.addForce(force_native_contacts1)
    
    def add_gaussian_nativec_individual(self,system,energy_attraction,u,d1,d2):
        r_cutoff_g=5.0*nanometer
        expr_gaussian_native_contacts="-Eatt*(A*exp(-B*r^2)+C*exp(-D*r^2))*step(r_ncg-r)"
        force_native_contacts=openmm.CustomNonbondedForce(expr_gaussian_native_contacts)
        force_native_contacts.addGlobalParameter("Eatt",energy_attraction)
        force_native_contacts.addGlobalParameter("A",4.6*kilojoule/mole)
        force_native_contacts.addGlobalParameter("B",10.0/(nanometer**2))
        force_native_contacts.addGlobalParameter("C",8.368*kilojoule/mole)
        force_native_contacts.addGlobalParameter("D",1.0/(nanometer**2))
        force_native_contacts.addGlobalParameter("r_ncg",r_cutoff_g)
    
        force_native_contacts.setNonbondedMethod(openmm.NonbondedForce.CutoffNonPeriodic)
        force_native_contacts.setCutoffDistance(r_cutoff_g)
        #Native contact pairs according to all-atom simulations
        ABCDpairs=setup_system.native_contact_list(d1,d2,u)
        ABCDpairs=np.asarray(ABCDpairs)-1
        number_native_contacts=len(ABCDpairs)
        for i in range(number_native_contacts):
            d1index=ABCDpairs[i][0]
            d2index=ABCDpairs[i][1]
            force_native_contacts.addInteractionGroup([d1index],[d2index])
        num_particles=system.getNumParticles()
        for i in range(num_particles):
            force_native_contacts.addParticle()
        system.addForce(force_native_contacts)


def main_simulation(energy_repulsion,energy_attraction,iter_number):
    print("HERE")
    start_time=time.time()
    dimer_list=['A2B2','A1B1','C1D1','C10D10','C23D23']
    #setup_system.create_system_pdb(dimer_list)
    pdbfile=home_dire+'/pentamer_box.pdb'
    u_system=mda.Universe(pdbfile)
    #define the system
    mdsteps=MDsteps(pdbfile)
    system=mdsteps.new_system(5)
    print(system.getForce)
    #add bonded forces
    print(dimer_list)
    mdsteps.create_harmonic_bonds(system,41840,dimer_list)
    print(system.getForces())
    print("HERE")
    #print(system.getForce(8).getNumBonds())

    #remove pre-assigned non-bonded forces (it tries to create LJ potential)
    mdsteps.remove_nonbonded_forces(system)
    system.removeForce(4)
    system.removeForce(4)

    #add all the forces we want: repulsive,attractive, anything else
    mdsteps.add_cosine_repulsion(system,energy_repulsion)
    #native contact forces
    mdsteps.add_gaussian_nativec(system,energy_attraction,u_system)
    #mdsteps.add_gaussian_nativec_individual(system,1.11*energy_attraction,u_system,'B1','C1')
    #mdsteps.add_gaussian_nativec_individual(system,1.3*energy_attraction,u_system,'C1','D10')
    #mdsteps.add_gaussian_nativec_individual(system,1.3*energy_attraction,u_system,'C10','D23')
    #mdsteps.add_gaussian_nativec_individual(system,1.3*energy_attraction,u_system,'C23','D1')
    #mdsteps.add_gaussian_nativec_individual(system,1.0*energy_attraction,u_system,'D1','B2')
 
    print(system.getForces())
    print("HERE")
    #define the integrator and simulation variables
    integrator=LangevinIntegrator(300*kelvin, 2/picosecond, 10.0*femtoseconds)
    #integrator=LangevinIntegrator(300*kelvin, 2/picosecond, 10*femtoseconds)
    print("HERE")
    integrator.setRandomNumberSeed(random.randint(0,1000))
    print(integrator.getRandomNumberSeed())
    #platform = Platform.getPlatformByName('CUDA') 
    simulation=Simulation(mdsteps.pdb.topology, system, integrator)
    simulation.context.setPositions(mdsteps.pdb.positions)
    ##########################MINIMIZATION#######################
    #minimization of initial structure
    simulation.minimizeEnergy(tolerance=0.1)
    simulation.saveState(f'{energy_attraction}/{iter_number}/minimized_pent{energy_repulsion}_{energy_attraction}.xml')
    #minpositions = simulation.context#.getState(getPositions=True).getPositions()
    #with open(f'minimized_{energy_repulsion}_{energy_attraction}.pdb', 'w') as f:
    #       PDBFile.writeFile(mdsteps.pdb.topology, minpositions, f)
   # #open minimized state or whatever state we want to continue our simulation from
   #
    simulation.loadState(f'{energy_attraction}/{iter_number}/minimized_pent{energy_repulsion}_{energy_attraction}.xml')
    simulation.reporters.append(DCDReporter(f'{energy_attraction}/{iter_number}/seg.dcd', 20000,enforcePeriodicBox=True))
    simulation.reporters.append(StateDataReporter(f'{energy_attraction}/{iter_number}/seg.csv', 20000, step=True, kineticEnergy=True, potentialEnergy=True, totalEnergy=True, temperature=True))
    simulation.reporters.append(CheckpointReporter(f'{energy_attraction}/{iter_number}/checkpnt_new.chk', 5000))
    print("HERE")
    #run the simulation for however many time steps
    simulation.step(500000000)
    #simulation.step(50000)
    simulation.saveState(f'{energy_attraction}/{iter_number}/final.xml')
    #finalpositions = simuflation.context.getState(getPositions=True).getPositions()
    #with open(f'final_{energy_repulsion}_{energy_attraction}.pdb', 'w') as f:
     #  PDBFile.writeFile(mdsteps.pdb.topology, finalpositions, f)
    final_time=time.time()
    print("{}mins".format((final_time-start_time)/60))
        
if __name__ == "__main__":
    reader = ParameterReader()
    params = reader.read_parameters()
    
    # Accessing parameters
    #psfile=params.psfpath
    #pdbfile=params.pdbpath
    energy_repulsion= params.Erepulsion
    energy_attraction= params.Enative
    iter_number= params.run
    #nrun= params.nrun
    main_simulation(energy_repulsion,energy_attraction,iter_number)




        
