#Write a function that creates a system with whatever concentration you want, however many dimers, defines interactions between dimers and within dimers
#Can place them at a certain distance of each with random orientation
## 1. Enumerate dimer types
## Using MDAnalysis is going to save my ass
import numpy as np
import MDAnalysis as mda
import random
import os 
import pandas as pd
home_dire=os.environ["WEST_SIM_ROOT"]
def move_dimer(u,space):
    #move to (0,0,0)
    #move to space with random orientation
    theta=random.uniform(0,2*np.pi)
    #print(theta)
    phi=random.uniform(0,np.pi)
    #print(phi)
    sphere_x=space*np.cos(theta)*np.sin(phi)
    sphere_y=space*np.sin(theta)*np.sin(phi)
    sphere_z=space*np.cos(phi)
    vec_trans=[sphere_x,sphere_y,sphere_z]
    #print(vec_trans)
    dimer_atoms= u.select_atoms("all")
    center = dimer_atoms.center_of_mass()
    dimer_atoms.positions=dimer_atoms.positions-center
    dimer_atoms.positions=dimer_atoms.positions+vec_trans
    return u

def separate_string(dname):
    letters = [char for char in dname if char.isalpha()]
    numbers = [char for char in dname if char.isdigit()]
    return letters,numbers

def change_segid(dimerold,dimernew):
    filenamenew='cg_'+dimernew+'_avg.pdb'
    print(filenamenew)
    filenameold='cg_'+dimerold+'_avg.pdb'
    print(filenameold)
    monomerold,numberold=separate_string(dimerold)
    #print(monomerold)
    #print(len(monomerold))
    monomernew, numbernew=separate_string(dimernew)
    m1old0=monomerold[0]+numberold[0]
    m1new0=monomernew[0]+numbernew[0]
    m1old1=monomerold[1]+numberold[1]
    m1new1=monomernew[1]+numbernew[1]
    with open(filenameold, 'r') as file:
        filedata = file.read()
    #print(filedata)
    filedata = filedata.replace(m1old0,m1new0)
    filedata = filedata.replace(m1old1,m1new1)


    with open(filenamenew, 'w') as file:
        file.write(filedata)



    


def create_system_pdb(dimer_list):
    print(dimer_list)
    combined_universe=mda.Universe('./cg_'+dimer_list[0]+'_avg.pdb')
    #combined_universe=move_dimer(combined_universe,0)
    #print((dimer_list[1:]))
    for j in range(len(dimer_list[1:])):
        print(dimer_list[j+1])
        monomer, number=separate_string(dimer_list[j+1])
        print(dimer_list[j+1])
        change_segid(monomer[0]+'1'+monomer[1]+'1',dimer_list[j+1])
        dimer_file=mda.Universe('cg_'+dimer_list[j+1]+'_avg.pdb')
        #print(dimer_file.atoms)
        #dimer_file=move_dimer(dimer_file,0*(j+1))
        combined_universe=mda.Merge(combined_universe.atoms,dimer_file.atoms)
    ag=combined_universe.select_atoms("name CA")
    ag.write('newsystem4.pdb')
    
    #print(combined_universe.atoms())    #name=name+dimer_list[j+1]+'_'
    dimer_bonds=[]
    for i in range(len(dimer_list)):
        monomer,number=separate_string(dimer_list[i])
        print(monomer)
        dimer_bonds_txt=np.loadtxt('cg_'+monomer[0]+monomer[1]+'_avg_connectivity.txt').T
        dimer_bond_number=len(dimer_bonds_txt[0][:])
        for bond_number in range(dimer_bond_number):
            dimer_bonds.append((298*i+int(dimer_bonds_txt[0][bond_number])-1,298*i+int(dimer_bonds_txt[1][bond_number])-1))
    #print(dimer_bonds)
    
    combined_universe.add_TopologyAttr('bonds',dimer_bonds)
    ag=combined_universe.select_atoms("name CA")
    #name=name+'connect'
    ag.write('penatmer_connect.pdb')
    
        


def native_contact_list(m1,m2,u):
    contactlist=pd.read_csv('./'+m1+'_'+m2+'_contacts.txt',sep='\t',header=None)
    native_contact_atoms=[]
    for i in range(len(contactlist)):
        mgroup=u.atoms[[]]
        ngroup=u.atoms[[]]
        res1=contactlist.iloc[i][1]
        sel1=['segid',m1,'and','resid',str(res1)]
        sel1=' '.join(sel1)
        print(sel1)
        mgroup=u.select_atoms(sel1)
        res2=contactlist.iloc[i][3]
        sel2=['segid',m2,'and','resid',str(res2)]
        sel2=' '.join(sel2)
        print(sel2)
        ngroup=u.select_atoms(sel2)
        #print(mgroup.atoms)
        print(mgroup.ids)
        print(ngroup.ids)
        native_contact_atoms.append((mgroup.ids[0],ngroup.ids[0]))
    return native_contact_atoms   


def contact_list_new(monomer_type,monomer_num,u,ubound):
    if(monomer_type)=='A':
        if((monomer_num-1)%5==0):
            sle_string='A'+str(monomer_num+4)
        else:
            sle_string='A'+str((monomer_num-1))
        #print(sle_string)
        sle=ubound.select_atoms('segid '+sle_string)
    if(monomer_type)=='B':
        sle=ubound.select_atoms(f"(around 9 segid B{monomer_num}) and not chainID D and not segid A{monomer_num}")
    if(monomer_type)=='C':
        sle=ubound.select_atoms(f"(around 9 segid C{monomer_num}) and not chainID A and not chainID B and not segid D{monomer_num}")
    if(monomer_type)=='D':
        sle=ubound.select_atoms(f"(around 9 segid D{monomer_num}) and not chainID C and not chainID A")
    if(len(list(set(sle.segids)))==0):
        return []
    m2=list(set(sle.segids))[0]
    m1=monomer_type+str(monomer_num)
    contactlist=pd.read_csv(home_dire+'/'+monomer_type+'_contacts.txt',sep='\t',header=None)
    native_contact_atoms=[]
    for i in range(len(contactlist)):
        mgroup=u.atoms[[]]
        ngroup=u.atoms[[]]
        res1=contactlist.iloc[i][1]
        sel1=['segid',m1,'and','resid',str(res1)]
        sel1=' '.join(sel1)
        print(sel1)
        mgroup=u.select_atoms(sel1)
        res2=contactlist.iloc[i][3]
        sel2=['segid',m2,'and','resid',str(res2)]
        sel2=' '.join(sel2)
        print(sel2)
        ngroup=u.select_atoms(sel2)
        #print(mgroup.atoms)
        print(mgroup.ids)
        print(ngroup.ids)
        if(len(mgroup.ids)!=0 and len(ngroup.ids)!=0):
            native_contact_atoms.append((mgroup.ids[0],ngroup.ids[0]))
        else:
            return []
        return native_contact_atoms  
