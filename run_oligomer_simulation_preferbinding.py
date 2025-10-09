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
#home_dire environment that has all relevant pdb, connectivity files
home_dire='/home/smriti/BigStorage/gnm_voth_model/decamer'
#define openmm forcefield just to initialize the system
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
        # Add more parameters as needed
       
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
        system.setDefaultPeriodicBoxVectors(Vec3(17.0*nanometer,0,0),Vec3(0,17.0*nanometer,0),Vec3(0,0,17.0*nanometer))
        return system
    def create_harmonic_bonds(self,system,bond_strength,dimer_list):
        #define bond strength and equilibrium bond distance
        harmonic_bond=HarmonicBondForce() 
        #Read bonded parameters and change the bond strength
        for i in range(len(dimer_list)):
            dimer_bonds_txt=np.loadtxt(home_dire+'/connect_files/cg_'+dimer_list[i]+'_connectivity.txt').T
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
        system.addForce(harmonic_bond)
                #return system
    def remove_nonbonded_forces(self,system):
        #remove all predefined non-bonded forces; we are defining our custom non-bonded forces
        #non-bonded force is 4th force in the default system definition
        nonbonded_og=system.getForce(4)
        for index in range(nonbonded_og.getNumParticles()):
            charge,sigma,epsilon=nonbonded_og.getParticleParameters(index)
            charge=0
            epsilon=0
            sigma=0
            nonbonded_og.setParticleParameters(index,charge,sigma,epsilon)
    def add_cosine_repulsion(self,system,energy_repulsion):
        #ensure exluded volume for the particles; parameters from voth paper
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
        #add cosine repulsion for all particles
        for i in range(num_particles):
            force_repulsion.addParticle()
        system.addForce(force_repulsion)
    
    def add_gaussian_nativec(self,system,energy_attraction,u,ubound):
        #key native contacts fron all-atom simulations
        ## TODO: fine tune A,B,C,D with all-atom simulation data
        r_cutoff_g=3.0*nanometer
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
        ABCDpairs=[]
        #add native contacts for each interface: A site(AA interface), B site(BC interface), C site(CD interface), D site(DB interface)
        for i in range(1,61):
            ABCDpairs=ABCDpairs+setup_system.contact_list_new('A',i,u,ubound)
            ABCDpairs=ABCDpairs+setup_system.contact_list_new('B',i,u,ubound)
            ABCDpairs=ABCDpairs+setup_system.contact_list_new('C',i,u,ubound)
            ABCDpairs=ABCDpairs+setup_system.contact_list_new('D',i,u,ubound)
        #print(ABCDpairs)
        ABCDpairs=np.asarray(ABCDpairs)-1
        number_native_contacts=len(ABCDpairs)
        #add each contact as an interaction group
        for i in range(number_native_contacts):
            d1index=ABCDpairs[i][0]
            d2index=ABCDpairs[i][1]
            force_native_contacts.addInteractionGroup([d1index],[d2index])
        num_particles=system.getNumParticles()
        for i in range(num_particles):
            force_native_contacts.addParticle()
        system.addForce(force_native_contacts)
    def gaussian_native_contactA(self,system,energy_attraction,u,ubound):
        r_cutoff_g=3.0*nanometer
        expr_gaussian_native_contactsA="-Eatt_contactA*(A*exp(-B*r^2)+C*exp(-D*r^2))*step(r_ncg-r)"
        force_native_Acontacts=openmm.CustomNonbondedForce(expr_gaussian_native_contactsA)
        force_native_Acontacts.addGlobalParameter("Eatt_contactA",1.17*energy_attraction)
        force_native_Acontacts.addGlobalParameter("A",4.6*kilojoule/mole)
        force_native_Acontacts.addGlobalParameter("B",10.0/(nanometer**2))
        force_native_Acontacts.addGlobalParameter("C",8.368*kilojoule/mole)
        force_native_Acontacts.addGlobalParameter("D",1.0/(nanometer**2))
        force_native_Acontacts.addGlobalParameter("r_ncg",r_cutoff_g)
    
        force_native_Acontacts.setNonbondedMethod(openmm.NonbondedForce.CutoffNonPeriodic)
        force_native_Acontacts.setCutoffDistance(r_cutoff_g)
        #Native contact pairs according to all-atom simulations
        Acontactpairs=[]
        #add native contacts for each interface: A site(AA interface), B site(BC interface), C site(CD interface), D site(DB interface)
        for i in range(1,61):
            Acontactpairs=Acontactpairs+setup_system.contact_list_new('A',i,u,ubound)
        Acontactpairs=np.asarray(Acontactpairs)-1
        number_native_contacts=len(Acontactpairs)
        #add each contact as an interaction group
        if(number_native_contacts==0):
            return
        else:
            for i in range(number_native_contacts):
                d1index=Acontactpairs[i][0]
                d2index=Acontactpairs[i][1]
                force_native_Acontacts.addInteractionGroup([d1index],[d2index])
            num_particles=system.getNumParticles()
            for i in range(num_particles):
                force_native_Acontacts.addParticle()
            system.addForce(force_native_Acontacts)

        #key native contacts fron all-atom simulations
    def gaussian_native_contactB(self,system,energy_attraction,u,ubound):
        r_cutoff_g=3.0*nanometer
        expr_gaussian_native_contactsB="-Eatt_contactB*(A*exp(-B*r^2)+C*exp(-D*r^2))*step(r_ncg-r)"
        force_native_Bcontacts=openmm.CustomNonbondedForce(expr_gaussian_native_contactsB)
        force_native_Bcontacts.addGlobalParameter("Eatt_contactB",1.11*energy_attraction)
        force_native_Bcontacts.addGlobalParameter("A",4.6*kilojoule/mole)
        force_native_Bcontacts.addGlobalParameter("B",10.0/(nanometer**2))
        force_native_Bcontacts.addGlobalParameter("C",8.368*kilojoule/mole)
        force_native_Bcontacts.addGlobalParameter("D",1.0/(nanometer**2))
        force_native_Bcontacts.addGlobalParameter("r_ncg",r_cutoff_g)

        force_native_Bcontacts.setNonbondedMethod(openmm.NonbondedForce.CutoffNonPeriodic)
        force_native_Bcontacts.setCutoffDistance(r_cutoff_g)
        #Native contact pairs according to all-atom simulations
        Bcontactpairs=[]
        #add native contacts for each interface: A site(AA interface), B site(BC interface), C site(CD interface), D site(DB interface)
        for i in range(1,61):
            Bcontactpairs=Bcontactpairs+setup_system.contact_list_new('B',i,u,ubound)
        Bcontactpairs=np.asarray(Bcontactpairs)-1
        number_native_contacts=len(Bcontactpairs)
        #add each contact as an interaction group
        if(number_native_contacts==0):
            return
        else:
            for i in range(number_native_contacts):
                d1index=Bcontactpairs[i][0]
                d2index=Bcontactpairs[i][1]
                force_native_Bcontacts.addInteractionGroup([d1index],[d2index])
            num_particles=system.getNumParticles()
            for i in range(num_particles):
                force_native_Bcontacts.addParticle()
            system.addForce(force_native_Bcontacts)

    def gaussian_native_contactC(self,system,energy_attraction,u,ubound):
        r_cutoff_g=3.0*nanometer
        expr_gaussian_native_contactsC="-Eatt_contactC*(A*exp(-B*r^2)+C*exp(-D*r^2))*step(r_ncg-r)"
        force_native_Ccontacts=openmm.CustomNonbondedForce(expr_gaussian_native_contactsC)
        force_native_Ccontacts.addGlobalParameter("Eatt_contactC",1.3*energy_attraction)
        force_native_Ccontacts.addGlobalParameter("A",4.6*kilojoule/mole)
        force_native_Ccontacts.addGlobalParameter("B",10.0/(nanometer**2))
        force_native_Ccontacts.addGlobalParameter("C",8.368*kilojoule/mole)
        force_native_Ccontacts.addGlobalParameter("D",1.0/(nanometer**2))
        force_native_Ccontacts.addGlobalParameter("r_ncg",r_cutoff_g)

        force_native_Ccontacts.setNonbondedMethod(openmm.NonbondedForce.CutoffNonPeriodic)
        force_native_Ccontacts.setCutoffDistance(r_cutoff_g)
        #Native contact pairs according to all-atom simulations
        Ccontactpairs=[]
        #add native contacts for each interface: A site(AA interface), B site(BC interface), C site(CD interface), D site(DB interface)
        for i in range(1,61):
            Ccontactpairs=Ccontactpairs+setup_system.contact_list_new('C',i,u,ubound)
        Ccontactpairs=np.asarray(Ccontactpairs)-1
        number_native_contacts=len(Ccontactpairs)
        #add each contact as an interaction grou
        if(number_native_contacts==0):
            return
        else:
            for i in range(number_native_contacts):
                d1index=Ccontactpairs[i][0]
                d2index=Ccontactpairs[i][1]
                force_native_Ccontacts.addInteractionGroup([d1index],[d2index])
            num_particles=system.getNumParticles()
            for i in range(num_particles):
                force_native_Ccontacts.addParticle()
            system.addForce(force_native_Ccontacts)
    def gaussian_native_contactD(self,system,energy_attraction,u,ubound):
        r_cutoff_g=3.0*nanometer
        expr_gaussian_native_contactsD="-Eatt_contactD*(A*exp(-B*r^2)+C*exp(-D*r^2))*step(r_ncg-r)"
        force_native_contactsD=openmm.CustomNonbondedForce(expr_gaussian_native_contactsD)
        force_native_contactsD.addGlobalParameter("Eatt_contactD",1.0*energy_attraction)
        force_native_contactsD.addGlobalParameter("A",4.6*kilojoule/mole)
        force_native_contactsD.addGlobalParameter("B",10.0/(nanometer**2))
        force_native_contactsD.addGlobalParameter("C",8.368*kilojoule/mole)
        force_native_contactsD.addGlobalParameter("D",1.0/(nanometer**2))
        force_native_contactsD.addGlobalParameter("r_ncg",r_cutoff_g)

        force_native_contactsD.setNonbondedMethod(openmm.NonbondedForce.CutoffNonPeriodic)
        force_native_contactsD.setCutoffDistance(r_cutoff_g)
        #Native contact pairs according to all-atom simulations
        Dcontactpairs=[]
        #add native contacts for each interface: A site(AA interface), B site(BC interface), C site(CD interface), D site(DB interface)
        for i in range(1,61):
            Dcontactpairs=Dcontactpairs+setup_system.contact_list_new('D',i,u,ubound)
        Dcontactpairs=np.asarray(Dcontactpairs)-1
        number_native_contacts=len(Dcontactpairs)
        #add each contact as an interaction group
        if(number_native_contacts==0):
            return
        else:
            for i in range(number_native_contacts):
                d1index=Dcontactpairs[i][0]
                d2index=Dcontactpairs[i][1]
                force_native_contactsD.addInteractionGroup([d1index],[d2index])
            num_particles=system.getNumParticles()
            for i in range(num_particles):
                force_native_contactsD.addParticle()
            system.addForce(force_native_contactsD)



def main_simulation(energy_repulsion,energy_attraction): 
    start_time=time.time()
    dimer_list=[]
    #number of dimers we want to simulate+ what kind and number(Example: Decamer has-(A1B1-A5B5,C1D1-C5D5))
    for i in range(1,6):
        dimer_list.append(f'A{i}B{i}')
        dimer_list.append(f'C{i}D{i}')
    #ubound has native contact definitions from bound state
    ubound=mda.Universe(home_dire+'/decamer_avg.pdb')
    #pdb file we want to simulate
    pdbfile=home_dire+'/decamer_sep.pdb'
    u_system=mda.Universe(pdbfile)
    #define the system
    mdsteps=MDsteps(pdbfile)
    system=mdsteps.new_system(5)
    #add bonded forces
    mdsteps.create_harmonic_bonds(system,41840,dimer_list)
    #remove pre-assigned non-bonded forces (it tries to create LJ potential)
    mdsteps.remove_nonbonded_forces(system)
    system.removeForce(4)
    
    #add all the forces we want: repulsive,attractive, anything else
    mdsteps.add_cosine_repulsion(system,energy_repulsion)
    #mdsteps.add_gaussian_nativec(system,energy_attraction,u_system,ubound)
    mdsteps.gaussian_native_contactA(system,energy_attraction,u_system,ubound)
    mdsteps.gaussian_native_contactB(system,energy_attraction,u_system,ubound)
    mdsteps.gaussian_native_contactC(system,energy_attraction,u_system,ubound)
    mdsteps.gaussian_native_contactD(system,energy_attraction,u_system,ubound)    
    #define the integrator and simulation variables
    integrator=LangevinIntegrator(300*kelvin, 2/picosecond, 10.0*femtoseconds)
    integrator.setRandomNumberSeed(random.randint(0,1000))
    #define platform might be useful for GPU simulations
    #platform = Platform.getPlatformByName('CUDA') 
    simulation=Simulation(mdsteps.pdb.topology, system, integrator)
    simulation.context.setPositions(mdsteps.pdb.positions)
    ##########################MINIMIZATION#######################
    #minimization of initial structure
    simulation.minimizeEnergy(tolerance=0.1)
    simulation.saveState(f'parent.xml')
    minpositions = simulation.context.getState(getPositions=True).getPositions()
    with open(f'minimized_{energy_repulsion}_{energy_attraction}.pdb', 'w') as f:
           PDBFile.writeFile(mdsteps.pdb.topology, minpositions, f)
    #open minimized state or whatever state we want to continue our simulation from
    #simulation.loadState(f'minimized_{energy_repulsion}_{energy_attraction}.xml') 
    ###############RUN SIMULATION##########################
    #if we are startinf from a minimized state or a previously saved state
    simulation.loadState(f'parent.xml')
    #reporters to save the simulation data and frequency
    simulation.reporters.append(DCDReporter('seg.dcd', 5000,enforcePeriodicBox=False))
    simulation.reporters.append(StateDataReporter('seg.csv', 5000, step=True, kineticEnergy=True, potentialEnergy=True, totalEnergy=True, temperature=True))
    #run the simulation for however many time steps
    simulation.step(10000)
    #save final state as xml which can be used for restarting the simulation
    simulation.saveState('seg.xml')
    finalpositions = simulation.context.getState(getPositions=True).getPositions()
    #save final state positions as a pdb file 
    with open(f'final_{energy_repulsion}_{energy_attraction}.pdb', 'w') as f:
       PDBFile.writeFile(mdsteps.pdb.topology, finalpositions, f)
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
    main_simulation(energy_repulsion,energy_attraction)




        
