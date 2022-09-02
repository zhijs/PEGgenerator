#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import warnings
warnings.filterwarnings('ignore')


# ### number of segments for the PEG molecules

# In[2]:


Nseg = 400


# ### box size

# In[3]:


Lx, Ly, Lz = 60, 60, 60


# ### load end-groups (head) and monomere up and down

# In[4]:


endpatch = np.loadtxt('/home/data/GROUP/yangzhi/project/gromac/surface/PEGgenerator-main/DATA/endpatch.dat')
monomer = np.loadtxt('/home/data/GROUP/yangzhi/project/gromac/surface/PEGgenerator-main/DATA/monomer.dat')
v = 0.28 # distance between 2 monomers
print('endpatch,,', endpatch)


# ### place number of segments

# In[5]:


atoms = np.zeros((30000,6))
cptatoms = 0
# place patch 1
for m in endpatch:
    atoms[cptatoms] = cptatoms+1, m[1], m[2], -m[3]-v, -m[4], m[5]
    cptatoms += 1
# place N monomers
for seg in range(Nseg):
    for m in monomer:
        atoms[cptatoms] = cptatoms+1, m[1], m[2], m[3]+seg*v*2, m[4], m[5]
        cptatoms += 1   
# place patch 2
for m in endpatch:
    atoms[cptatoms] = cptatoms+1, m[1], m[2], m[3]+(2*seg)*v+v, m[4], m[5]
    cptatoms += 1
atoms = atoms[:cptatoms]
car = atoms[atoms.T[1] == 1]
hyd = atoms[(atoms.T[1] == 3) | (atoms.T[1] == 5)]
oxy = atoms[(atoms.T[1] == 2) | (atoms.T[1] == 4)]


# ### center PEG in box

# In[6]:


atoms.T[3] -= np.mean(atoms.T[3])
atoms.T[3] += Lx/2
atoms.T[4] -= np.mean(atoms.T[4])
atoms.T[4] += Ly/2
atoms.T[5] -= np.mean(atoms.T[5])
atoms.T[5] += Lz/2


# ### estimate molar mass

# In[7]:


molmass = len(car)*12+len(oxy)*16+len(hyd)*1
print('PEG - '+str(molmass)+' g/mol')


# ### add bonds

# In[8]:


bonds = np.zeros((30000,2))
cptbonds = 0
# carbon - carbon bonds between monomers
for idx0 in np.int32(car.T[0][:-1:2]):
    idx1 = np.int32(car.T[0][np.where(car.T[0] == idx0)[0][0]+1])
    if idx0<idx1:
        bonds[cptbonds] = idx0, idx1
    else:
        bonds[cptbonds] = idx1, idx0
    cptbonds += 1
# carbon - oxygen bonds
xyz = car.T[3:].T
for n0 in range(len(oxy)):
    xyz0 = oxy[n0][3:]
    idx0 = np.int32(oxy[n0][0])
    d = np.sqrt((xyz.T[0]-xyz0[0])**2+(xyz.T[1]-xyz0[1])**2+(xyz.T[2]-xyz0[2])**2)
    where = np.where((d > 0) & (d < 0.15))
    for w in where[0]:
        idx1 = np.int32(car[w][0])
        if idx0<idx1:
            bonds[cptbonds] = idx0, idx1
        else:
            bonds[cptbonds] = idx1, idx0
        cptbonds += 1
# carbon - hydrogen bonds
xyz = car.T[3:].T
for n0 in range(len(hyd)):
    xyz0 = hyd[n0][3:]
    idx0 = np.int32(hyd[n0][0])
    d = np.sqrt((xyz.T[0]-xyz0[0])**2+(xyz.T[1]-xyz0[1])**2+(xyz.T[2]-xyz0[2])**2)
    where = np.where((d > 0) & (d < 0.11))[0]
    if where.shape == (1,):
        idx1 = car[where][0][0]
        if idx0<idx1:
            bonds[cptbonds] = idx0, idx1
        else:
            bonds[cptbonds] = idx1, idx0
        cptbonds += 1      
# oxygen - hydrogen bonds
xyz = oxy.T[3:].T
for n0 in range(len(hyd)):
    xyz0 = hyd[n0][3:]
    idx0 = np.int32(hyd[n0][0])
    d = np.sqrt((xyz.T[0]-xyz0[0])**2+(xyz.T[1]-xyz0[1])**2+(xyz.T[2]-xyz0[2])**2)
    where = np.where((d > 0) & (d < 0.11))[0]
    if where.shape == (1,):
        idx1 = oxy[where][0][0]
        if idx0<idx1:
            bonds[cptbonds] = idx0, idx1
        else:
            bonds[cptbonds] = idx1, idx0
        cptbonds += 1       
# remove excess lines and reorder
bonds = bonds[:cptbonds]
bonds = bonds[bonds[:, 0].argsort()]


# ### calculate angles

# In[9]:


angles = np.zeros((30000,3))
cptangles = 0
bonded_a = np.append(bonds.T[0],bonds.T[1])
for a in atoms:
    ida = np.int32(a[0])
    tpa = np.int32(atoms[atoms.T[0] == ida].T[1])[0]
    occurence = np.sum(bonded_a == ida)
    if occurence > 1: # the atom has 2 or more atoms
        id_neighbors = np.unique(bonds[(bonds.T[0] == ida) | (bonds.T[1] == ida)].T[:2].T)
        for idb in id_neighbors:
            for idc in id_neighbors:
                if (idb != ida) & (idc != ida) & (idb < idc): # avoid counting same angle twice
                    angles[cptangles] = idb, ida, idc
                    cptangles += 1       
angles = angles[:cptangles]


# ## calculate dihedrals

# In[10]:


dihedrals = np.zeros((30000,4))
cptdihedrals = 0
central_angled_a = angles.T[1]
edge_angled_a = np.append(angles.T[0],angles.T[2])
for a in atoms:
    ida = np.int32(a[0])
    tpa = np.int32(atoms[atoms.T[0] == ida].T[1])[0]
    if (tpa == 1) | (tpa == 2) | (tpa == 4): # ignore hydrogen
        id_first_neighbor = np.unique(angles[(angles.T[1] == ida)].T[:3].T)
        id_first_neighbor = id_first_neighbor[id_first_neighbor != ida]
        for idb in id_first_neighbor:
            id_second_neighbor = np.unique(angles[(angles.T[1] == idb)].T[:3].T)
            if len(id_second_neighbor)>0:
                id_second_neighbor = id_second_neighbor[id_second_neighbor != idb]
                id_second_neighbor = id_second_neighbor[id_second_neighbor != ida]
                for idc in id_first_neighbor:
                    if idc != idb:
                        for ide in id_second_neighbor:
                            tpc = np.int32(atoms[atoms.T[0] == idc].T[1])[0]
                            tpe = np.int32(atoms[atoms.T[0] == ide].T[1])[0]
                            if (ida < idb) & (tpc != 3) & (tpe != 3) : 
                                dihedrals[cptdihedrals] = idc, ida, idb, ide
                                cptdihedrals += 1
dihedrals = dihedrals[:cptdihedrals]


# ### write conf file

# In[11]:


f = open('conf.gro', 'w')
f.write('PEG SYSTEM\n')
f.write(str(cptatoms)+'\n')
nc, no, nh = 0,0,0
for n in range(cptatoms):
    f.write("{: >5}".format(str(1))) # residue number (5 positions, integer) 
    f.write("{: >5}".format('PEG')) # residue name (5 characters)
    if (atoms.T[1][n] == 3) | (atoms.T[1][n] == 5):
        nh += 1
        f.write("{: >5}".format('H'+str(nh))) # atom name (5 characters) 
    elif (atoms.T[1][n] == 2) | (atoms.T[1][n] == 4):
        no += 1
        f.write("{: >5}".format('O'+str(no))) # atom name (5 characters) 
    elif atoms.T[1][n] == 1:
        nc += 1
        f.write("{: >5}".format('C'+str(nc))) # atom name (5 characters) 
    else:
        print('extra atoms')
    f.write("{: >5}".format(str(np.int32(n+1)))) # atom number (5 positions, integer)
    f.write("{: >8}".format(str("{:.3f}".format(atoms[n][3])))) # position (in nm, x y z in 3 columns, each 8 positions with 3 decimal places)
    f.write("{: >8}".format(str("{:.3f}".format(atoms[n][4])))) # position (in nm, x y z in 3 columns, each 8 positions with 3 decimal places) 
    f.write("{: >8}".format(str("{:.3f}".format(atoms[n][5])))) # position (in nm, x y z in 3 columns, each 8 positions with 3 decimal places) 
    f.write("\n")
f.write("{: >10}".format(str("{:.5f}".format(Lx))))
f.write("{: >10}".format(str("{:.5f}".format(Ly))))
f.write("{: >10}".format(str("{:.5f}".format(Lz))))
f.close()


# ### write itp file

# In[12]:


f = open('peg.itp', 'w')
f.write('[ moleculetype ]\n')
f.write('PEG   2\n\n')
f.write('[ atoms ]\n')
nc = 0
no = 0
nh = 0
for n in range(cptatoms):
    f.write("{: >5}".format(str(n+1))) # atom number
    if atoms.T[1][n] == 1:
        f.write("{: >8}".format('CC32A'))
    elif atoms.T[1][n] == 2:
        f.write("{: >8}".format('OC30A'))
    elif atoms.T[1][n] == 3:
        f.write("{: >8}".format('HCA2'))
    elif atoms.T[1][n] == 4:
        f.write("{: >8}".format('OC311'))
    elif atoms.T[1][n] == 5:
        f.write("{: >8}".format('HCP1'))
    else:
        print('extra atoms')    
    f.write("{: >8}".format(str(1))) # residue number
    f.write("{: >8}".format('PEG')) # residue number
    if atoms.T[1][n] == 1:
        nc += 1
        f.write("{: >8}".format('C'+str(nc))) # atom name
    elif (atoms.T[1][n] == 3) | (atoms.T[1][n] == 5):
        nh += 1
        f.write("{: >8}".format('H'+str(nh))) # atom name
    elif (atoms.T[1][n] == 2) | (atoms.T[1][n] == 4):
        no += 1
        f.write("{: >8}".format('O'+str(no))) # atom name
    f.write("{: >8}".format(str(np.int32(n+1))))
    f.write("{: >8}".format(str("{:.3f}".format(atoms.T[2][n]))))
    if atoms.T[1][n] == 1:
        f.write("{: >8}".format(str("{:.3f}".format(12.011))))
    elif (atoms.T[1][n] == 3) | (atoms.T[1][n] == 5):
        f.write("{: >8}".format(str("{:.3f}".format(1.008))))    
    elif (atoms.T[1][n] == 2) | (atoms.T[1][n] == 4):
        f.write("{: >8}".format(str("{:.3f}".format(15.9994)))) 
    f.write("\n") 
f.write("\n")  
f.write('[ bonds ]\n')  
for n in range(cptbonds):
    f.write("{: >5} ".format(str(np.int32(bonds[n][0]))))
    f.write("{: >5} ".format(str(np.int32(bonds[n][1]))))
    f.write("{: >5} ".format(str(np.int32(1))))
    f.write("\n")
f.write("\n")  
f.write('[ angles ]\n')  
for n in range(cptangles):
    f.write("{: >5} ".format(str(np.int32(angles[n][0]))))
    f.write("{: >5} ".format(str(np.int32(angles[n][1]))))
    f.write("{: >5} ".format(str(np.int32(angles[n][2]))))
    f.write("{: >5} ".format(str(np.int32(5))))
    f.write("\n")
f.write("\n")  
f.write('[ dihedrals ]\n')  
for n in range(cptdihedrals):
    f.write("{: >5} ".format(str(np.int32(dihedrals[n][0]))))
    f.write("{: >5} ".format(str(np.int32(dihedrals[n][1]))))
    f.write("{: >5} ".format(str(np.int32(dihedrals[n][2]))))
    f.write("{: >5} ".format(str(np.int32(dihedrals[n][3]))))
    f.write("{: >5} ".format(str(np.int32(9))))
    f.write("\n")
f.close()

