from copy import deepcopy
import pickle
from rdkit import Chem
import re
from tqdm import tqdm
import numpy as np

def get_atom_position(frag,at):
    return np.array(list(frag.GetConformer().GetAtomPosition(at)))

def _dihedral(p):
    """Praxeolitic formula
    1 sqrt, 1 cross product"""
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]

    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))

def _spiro_dihedral(p):
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]
    p4 = p[4]

    b0 = p0 - p2
    b1 = p1 - p2
    b2 = p3 - p2
    b3 = p4 - p2

    # c0 = (b0+b1)/2
    # c1 = (b2+b3)/2
    # c0/=np.linalg.norm(c0)
    # c1/=np.linalg.norm(c1)
    # degree2 = np.degrees(np.arccos(np.dot(c0,c1)))
   
    b0xb1 = np.cross(b0, b1)
    b0xb1 /= np.linalg.norm(b0xb1)
    b2xb3 = np.cross(b2, b3)
    b2xb3 /= np.linalg.norm(b0xb1)
    degree1 = np.degrees(np.arccos(np.dot(b0xb1,b2xb3)))
    # return np.degrees(np.arctan2(y, x))
    return degree1
    # return np.degrees(np.arctan2(y, x))

def calc_chain_dihedral(frag1,frag2,link_atoms1,link_atoms2):
    if len(frag1.GetAtoms())<=2 or len(frag2.GetAtoms())<=2:
        return 0.0
    p1 = get_atom_position(frag1, link_atoms1[0])
    p3 = get_atom_position(frag1, link_atoms1[2])
    p2 = get_atom_position(frag2, link_atoms2[0])
    p4 = get_atom_position(frag2, link_atoms2[2])
    return _dihedral([p3,p2,p1,p4])


def calc_plain_dihedral(frag1,frag2,atoms1,atoms2):
        # print(atoms1,atoms2)
        p1 = (get_atom_position(frag1, atoms1[2]) + get_atom_position(frag1,  atoms1[3]))/2
        p4 = (get_atom_position(frag2, atoms2[2]) + get_atom_position(frag2,  atoms2[3]))/2
        p2 =  get_atom_position(frag1, atoms1[0])
        p3 =  get_atom_position(frag1, atoms1[1])
        return _dihedral([p1,p2,p3,p4])
 
    
def calc_spiro_dihedral(frag1,frag2,atoms1,atoms2):
        
        p1 = get_atom_position(frag1, atoms1[1])
        p2 = get_atom_position(frag1, atoms1[2])
        p4 = get_atom_position(frag2, atoms2[1])
        p5 = get_atom_position(frag2, atoms2[2])
        p3 = get_atom_position(frag1, atoms1[0])
        return _spiro_dihedral([p1,p2,p3,p4,p5])
    
def get_line_transform_matrix(a1,a2,b1,b2):
    # attention! a1 == b1 a2==b2

    l1 = b2-b1
    length1 = np.linalg.norm(l1)
    l2 = a2-a1
    length2 = np.linalg.norm(l2)
    costheta = np.dot(l1,l2)/(length1*length2)
    if costheta >1:
        costheta =1
    if costheta <-1:
        costheta = -1
    theta = np.arccos(costheta)
    k = np.cross(l1,l2)
    k = k / np.linalg.norm(k)
    k = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    r = np.identity(3) + np.sin(theta) *k+ (1-costheta)* np.dot(k,k)
    r = np.array([[r[0][0],r[0][1],r[0][2],0],[r[1][0],r[1][1],r[1][2],0],[r[2][0],r[2][1],r[2][2],0],[0,0,0,1]])
    t = np.array([a1[0],a1[1],a1[2],1]) - np.dot(r,np.array([b1[0],b1[1],b1[2],1]))
    t = np.array([[1,0,0,t[0]],[0,1,0,t[1]],[0,0,1,t[2]],[0,0,0,1]])
    r = np.matmul(t,r)
    return  r
