from utils.fragment import get_clean_smi, swap_atom_order, edge_bfs, wash_frag,reorder_frag
from utils.calc_dihedral import get_atom_position, get_line_transform_matrix
from utils.calc_dihedral import calc_chain_dihedral, calc_plain_dihedral, calc_spiro_dihedral

from rdkit.Chem.rdMolAlign import AlignMol
from rdkit.Chem import rdMolTransforms as rT
from rdkit.Geometry import Point3D
from rdkit import Chem
from copy import deepcopy
from data.vocabulary import FragVocab
from segmentor.frag_seg import FragSeg
import numpy as np
import pickle

class FragProcessor:
    def __init__(self,segmentor = None ,vocab = None,angle_interval = 5):
        super(FragProcessor, self).__init__()
        self.sg = segmentor
        self.vocab = vocab
        self.angle_interval = angle_interval

    def bulid(self,rd_mols,save_path = 'geom-drugs'):
        self.vocab = FragVocab()
        vocab_file,sg_hit_vocab =self.vocab.build_vocab(rd_mols)
        self.vocab.load_vocab(vocab_file)
        self.sg = FragSeg(hit_vocab=sg_hit_vocab)
        pickle.dump(vocab_file,open(f'{save_path}/vocab2.pkl','wb'))
        pickle.dump(sg_hit_vocab,open(f'{save_path}/hit2.pkl','wb'))
        
    def sanitize(self,mol):
        # rd_mol = mol
        try:
            smiles = Chem.MolToSmiles(mol)
                # print(111)
            if "." in smiles:
                return None
            Chem.SanitizeMol(mol) 
                            
            mol = Chem.rdmolops.RemoveAllHs(mol, sanitize=True)
            assert mol.GetNumHeavyAtoms() == mol.GetNumAtoms(), "cannot remove all Hs for %s" % smiles
        except Exception:
            return None
        return mol
   
    def fragmentize_sanitized_mol(self,mol):
        frags,_,graph_info = self.sg.fragmenize(mol,return_graph = True)
        # for x in frags:
        #     print(get_clean_smi(x))
        edge_index = graph_info['edge_index']
        edge = list(zip(edge_index[0],edge_index[1]))
        nbrs = {k:[] for k in range(len(frags))}
        for e in edge:
            nbrs[e[0]].append(e[1])
            nbrs[e[1]].append(e[0])

        iface_atoms = graph_info['iface_atoms']
        iface_atoms = list(zip(iface_atoms[0],iface_atoms[1]))
        iface_types = graph_info['iface_type']

        d2_index = []
        d3_index = []
        
        for frag in frags:
            frag = reorder_frag(wash_frag(frag))
            frag_smi = get_clean_smi(frag)
            frag_index = self.vocab.SmilesToFrag(frag_smi)
            d2_index.append(frag_index)
            best_rmsd = 1000
            best_idx = -1
            # frag2 = deepcopy(frag)
            # frag2 = reorder_frag(wash_frag(frag2))
            for idx,frag_conf in self.vocab.FragToMols(frag_index):
                rmsd = AlignMol(frag,frag_conf,atomMap = [(i,i) for i in range(len(frag.GetAtoms()))])
                if rmsd< best_rmsd:
                    best_rmsd = rmsd
                    best_idx = idx
            d3_index.append(best_idx)
        # except Exception as e:
        #     print(e)
        #     return None
        iface_index = []
        dihedrals = []
        for e,l,t in zip(edge,iface_atoms,iface_types):
            frag1_index = d2_index[e[0]]
            frag2_index = d2_index[e[1]]
            idx1 = self.vocab.FragAtomToIface(frag1_index,l[0])
            idx2 = self.vocab.FragAtomToIface(frag2_index,l[1])
            iface_index.append([idx1,idx2])
            # iface_type = iface_types[i]

            frag1 = frags[e[0]]
            frag2 = frags[e[1]]
            iface1 = self.vocab.IfaceToDihAtoms(frag1_index,idx1)
            iface2 = self.vocab.IfaceToDihAtoms(frag2_index,idx2)
            # if self.dihedral_order(e) < 0:
            #     print('reorder',e)
            #     frag1,frag2 = frag2,frag1
            #     iface1,iface2 = iface2,iface1
            if t == 0:
                calc_func = calc_chain_dihedral
            elif t==1:
                calc_func = calc_spiro_dihedral
            else:
                if self.dihedral_order(e,nbrs,d2_index) < 0:
                    # print('reorder',e)
                    frag1,frag2 = frag2,frag1
                    iface1,iface2 = iface2,iface1
                calc_func = calc_plain_dihedral
                
            

            dihedrals.append(float(format(calc_func(frag1,frag2,iface1,iface2),".2f")))

        return {
                "frag_index":d2_index,
                "conf_index":d3_index,
                "edge_index":edge_index,
                "iface_index":iface_index,
                "iface_types":iface_types,
                "dihedrals":dihedrals,
            }

    
    def assemble_mol(self,conf_index,dihedrals,edge_index,iface_index,iface_types):

        # done = set()
        if not conf_index:
            return None
        # done_edge_index =set()
        if isinstance(dihedrals[0],int):
            dihedrals = [self.idx2angle(i) for i in dihedrals]
        frag_index = [self.vocab.ConfToFrag(i) for i in conf_index]
        iface,dih,itype = {},{},{}
        nbrs = {k:[] for k in range(len(conf_index))}
        edge_index = list(zip(edge_index[0],edge_index[1]))

        for e,l,d,t in zip(edge_index,iface_index,dihedrals,iface_types):
            iface[e],dih[e],itype[e] = l,d,t
            iface[(e[1],e[0])],dih[(e[1],e[0])],itype[(e[1],e[0])] = (l[1],l[0]),d,t
            nbrs[e[0]].append(e[1])
            nbrs[e[1]].append(e[0])
        
        # frags = [Chem.RWMol(deepcopy(self.vocab.ConfToMol(i))) for i in conf_index]
        mol = Chem.RWMol(deepcopy(self.vocab.ConfToMol(conf_index[0])))
        self.atom_map = {0:list(range(len(mol.GetAtoms())))}
        visited = [0]
        
        edges = edge_bfs(edge_index,iface_types,source=0)
        # print(edges)
        for e in edges:
            
            conf1_index = conf_index[e[0]]
            conf2_index = conf_index[e[1]]
            iface1_index,iface2_index = iface[e]
            if e[1] not in visited:
                # frag_conf = deepcopy(self.vocab.ConfToMol(self.d3_index[e[1]]))
                
                
                frag = Chem.RWMol(deepcopy(self.vocab.ConfToMol(conf2_index)))
                mol_iface = [self.atom_map[e[0]][at] for at in self.vocab.ConfToDihAtoms(conf1_index,iface1_index)]
                try:
                    frag_iface = self.vocab.ConfToDihAtoms(conf2_index,iface2_index)
                except:
                    print('error',iface_index)
                dihedral = dih[e]
                if itype[e] == 0:
                    self.asm_chain_frag(mol = mol,
                                        frag = frag,
                                        edge=e,
                                        mol_iface = mol_iface,
                                        frag_iface = frag_iface,
                                        dihedral = dihedral
                                    )
                elif itype[e] in (2,3):
                    self.asm_plain_frag(mol = mol,
                                        frag = frag,
                                        edge=e,
                                        mol_iface = mol_iface,
                                        frag_iface = frag_iface,
                                        dihedral = self.dihedral_order(e,nbrs,frag_index) * dihedral,
                                        reflection=False if itype[e] == 2 else True)
                    # print(Chem.MolToSmiles(mol))
                                        
                elif itype[e] == 1:
                    self.asm_spiro_frag(mol = mol,
                                        frag = frag,
                                        edge=e,
                                        mol_iface = mol_iface,
                                        frag_iface = frag_iface,
                                        dihedral = dihedral,
                                        )

                visited.append(e[1])
            else:
                if itype[e] in (2,3):
                    frag1_iface = [self.atom_map[e[0]][at] for at in self.vocab.ConfToDihAtoms(conf1_index,iface1_index)]
                    frag2_iface = [self.atom_map[e[1]][at] for at in self.vocab.ConfToDihAtoms(conf2_index,iface2_index)]
                    self.asm_plain_pah(mol = mol,
                                        edge=e, 
                                        frag1_iface = frag1_iface,
                                        frag2_iface = frag2_iface,
                                        reflection=False if itype[e] == 2 else True)
            # print(Chem.MolToSmiles(mol))
        #some sanitization
        try:
            for m in list(Chem.rdmolops.GetMolFrags(mol, asMols=True)):
                if len(m.GetAtoms())>1:
                    return m
        except Exception:
            print(Exception)
            return None

        return mol

    def dihedral_order(self,edge,nbrs,d2_index):
        frag1_index = d2_index[edge[0]]
        frag2_index = d2_index[edge[1]]
        if frag1_index > frag2_index:
            return -1
        elif frag1_index == frag2_index:
            frag1_nbr = sum( [d2_index[nbr] for nbr in nbrs[edge[0]]])
            frag2_nbr = sum( [d2_index[nbr] for nbr in nbrs[edge[1]]])
            if frag1_nbr > frag2_nbr:
                return -1
        return 1


    def asm_chain_frag(self,mol,frag,edge,mol_iface,frag_iface,dihedral):
        conf = mol.GetConformer(0)
        f_conf = frag.GetConformer(0)
        trans_matrix = get_line_transform_matrix(get_atom_position(mol, mol_iface[1]),
                                                    get_atom_position(mol, mol_iface[0]),
                                                    get_atom_position(frag, frag_iface[0]),
                                                    get_atom_position(frag, frag_iface[1])
                                                    )
        rT.TransformConformer(f_conf,trans_matrix)
        bond_type = frag.GetBondBetweenAtoms(frag_iface[0],frag_iface[1]).GetBondType()
        mol.RemoveBond(mol_iface[0],mol_iface[1])
        frag.RemoveBond(frag_iface[0],frag_iface[1])
        self.atom_map[edge[1]] = []
        for atom in frag.GetAtoms():
            at_idx = atom.GetIdx()
            new_at_idx = mol.AddAtom(atom)
            self.atom_map[edge[1]].append(new_at_idx)
            x,y,z = f_conf.GetAtomPosition(at_idx)
            # AtomPosition(nconf.Setew_at_idx,Point3D(x,y,z))
            conf.SetAtomPosition(new_at_idx,Point3D(x,y,z))
        for bond in frag.GetBonds():
            mol.AddBond(self.atom_map[edge[1]][bond.GetBeginAtomIdx()],
                        self.atom_map[edge[1]][bond.GetEndAtomIdx()],
                        order= bond.GetBondType())
        # mol.RemoveAtom()
        frag_iface = [self.atom_map[edge[1]][at] for at in frag_iface]
        mol.AddBond(mol_iface[1],frag_iface[1],order=bond_type)
        Chem.SanitizeMol(mol,sanitizeOps=Chem.SanitizeFlags.SANITIZE_SYMMRINGS)
        if len(mol_iface)==3 and len(frag_iface)==3:

            rT.SetDihedralDeg(conf,
                              mol_iface[2],
                              mol_iface[1],
                              frag_iface[1],
                              frag_iface[2],
                              dihedral)

    def asm_spiro_frag(self,mol,frag,edge,mol_iface,frag_iface,dihedral):
        conf = mol.GetConformer(0)
        f_conf = frag.GetConformer(0)
        ori_frag_atom = frag_iface[0]
        mol_mid =  (get_atom_position(mol, mol_iface[1]) + get_atom_position(mol, mol_iface[2]))/2
        # mid = 2*get_atom_position(mol, mol_iface[0]) - mol_mid
        frag_mid = (get_atom_position(frag, frag_iface[1]) + get_atom_position(frag, frag_iface[2]))/2
        # t = get_atom_position(frag, frag_iface[0]) - frag_mid
        length = np.linalg.norm(get_atom_position(frag, frag_iface[0]) - frag_mid)
        t = get_atom_position(mol, mol_iface[0]) - mol_mid
        mid = t/np.linalg.norm(t)*length + get_atom_position(mol, mol_iface[0])
        # x2,y2,z2 = conf.GetAtomPosition(mol_iface[2])
        # x3,y3,z3 = conf.GetAtomPosition(mol_iface[0])
        trans_matrix = get_line_transform_matrix(get_atom_position(mol, mol_iface[0]),
                                                    mid,
                                                    get_atom_position(frag, frag_iface[0]),
                                                    frag_mid
                                                    )
        rT.TransformConformer(f_conf,trans_matrix)
        # print(mid,(get_atom_position(frag, frag_iface[1]) + get_atom_position(frag, frag_iface[2]))/2)
        # print(get_atom_position(mol, mol_iface[0]),get_atom_position(frag, frag_iface[0]))
        # print(dp(mol_mid))
        mol_remove_bonds = {}
        frag_remove_bonds = {}

        for nbr in frag.GetAtomWithIdx(frag_iface[0]).GetNeighbors():
            nbr_idx = nbr.GetIdx()
            bt = frag.GetBondBetweenAtoms(frag_iface[0],nbr_idx).GetBondType()
            frag_remove_bonds[(frag_iface[0],nbr_idx)] = bt
            frag.RemoveBond(frag_iface[0],nbr_idx)
        # print(Chem.MolToSmiles(frag))

        for nbr in mol.GetAtomWithIdx(mol_iface[0]).GetNeighbors():
            nbr_idx = nbr.GetIdx()
            bt = mol.GetBondBetweenAtoms(mol_iface[0],nbr_idx).GetBondType()
            mol_remove_bonds[(mol_iface[0],nbr_idx)] = bt
            mol.RemoveBond(mol_iface[0],nbr_idx)
        # print(Chem.MolToSmiles(mol))
        self.atom_map[edge[1]] = []

        for atom in frag.GetAtoms():
            at_idx = atom.GetIdx()
            new_at_idx = mol.AddAtom(atom)
            self.atom_map[edge[1]].append(new_at_idx)
            x,y,z = f_conf.GetAtomPosition(at_idx)
            conf.SetAtomPosition(new_at_idx,Point3D(x,y,z))
        # print(self.atom_map)
        for bond in frag.GetBonds():
            mol.AddBond(self.atom_map[edge[1]][bond.GetBeginAtomIdx()],
                        self.atom_map[edge[1]][bond.GetEndAtomIdx()],
                        order= bond.GetBondType())
        frag_iface = [self.atom_map[edge[1]][at] for at in frag_iface]

        dummy1= mol.AddAtom(Chem.Atom(80))
        dummy2= mol.AddAtom(Chem.Atom(80))
        
        
        

        conf.SetAtomPosition(dummy1,Point3D(mol_mid[0],mol_mid[1],mol_mid[2]))
        conf.SetAtomPosition(dummy2,Point3D(mid[0],mid[1],mid[2]))
        for e in mol_remove_bonds:
            mol.AddBond(e[1],dummy1)
        for e in frag_remove_bonds:
            mol.AddBond(dummy2,self.atom_map[edge[1]][e[1]])
        mol.AddBond(dummy1,dummy2)

        Chem.SanitizeMol(mol,sanitizeOps=Chem.SanitizeFlags.SANITIZE_SYMMRINGS)
        rT.SetDihedralDeg(conf,mol_iface[1],dummy1,dummy2,frag_iface[1],dihedral)

        for e in mol_remove_bonds:
            mol.RemoveBond(e[1],dummy1)
        for e in frag_remove_bonds:
            mol.RemoveBond(dummy2,self.atom_map[edge[1]][e[1]])
        # print(Chem.MolToSmiles(mol))
        for bond in frag_remove_bonds:
            mol.AddBond(mol_iface[0],self.atom_map[edge[1]][bond[1]],order=frag_remove_bonds[bond])
        for bond in mol_remove_bonds:
            mol.AddBond(mol_iface[0],bond[1],order=mol_remove_bonds[bond])

        mol.GetAtomWithIdx(frag_iface[0]).SetIsAromatic(0)
        self.atom_map[edge[1]][ori_frag_atom] = mol_iface[0]
        mol.GetAtomWithIdx(mol_iface[0]).SetNumExplicitHs(0)
        # print(Chem.MolToSmiles(mol))
        # mol.GetAtomWithIdx(mol_iface[1]).SetNumExplicitHs(max(mol.GetAtomWithIdx(mol_iface[1]).GetNumExplicitHs()-1,0))
            
        
    

    def asm_plain_frag(self,mol,frag,edge,mol_iface,frag_iface,dihedral,reflection = False):
        conf = mol.GetConformer(0)
        f_conf = frag.GetConformer(0)
        ori_frag_atom0 = frag_iface[0]
        ori_frag_atom1 = frag_iface[1]
        if reflection:
            frag_iface = swap_atom_order(frag_iface)
            ori_frag_atom0,ori_frag_atom1 = ori_frag_atom1,ori_frag_atom0
        # print(mol_iface,frag_iface)
        # print(self.atom_map)
        #mol_at0 == frag_at1 mol_at1 = frag_at0
        # print('frag_init',Chem.MolToSmiles(frag))
        trans_matrix = get_line_transform_matrix(get_atom_position(mol, mol_iface[0]),
                                                    get_atom_position(mol, mol_iface[1]),
                                                    get_atom_position(frag, frag_iface[0]),
                                                    get_atom_position(frag, frag_iface[1])
                                                    )
        rT.TransformConformer(f_conf,trans_matrix)
        
        mol_remove_bonds = {}
        for atom1,atom2 in [(mol_iface[0],mol_iface[1]),(mol_iface[1],mol_iface[0])]:
            for nbr in mol.GetAtomWithIdx(atom1).GetNeighbors():
                nbr_idx = nbr.GetIdx()
                if nbr_idx!= atom2:
                    bt = mol.GetBondBetweenAtoms(atom1,nbr_idx).GetBondType()
                    mol_remove_bonds[(atom1,nbr_idx)] = bt
                    mol.RemoveBond(atom1,nbr_idx)

        frag_remove_bonds = {}
        for atom1,atom2 in [(frag_iface[0],frag_iface[1]),(frag_iface[1],frag_iface[0])]:
            for nbr in frag.GetAtomWithIdx(atom1).GetNeighbors():
                nbr_idx = nbr.GetIdx()
                if nbr_idx!= atom2:
                    bt = frag.GetBondBetweenAtoms(atom1,nbr_idx).GetBondType()
                    frag_remove_bonds[(atom1,nbr_idx)] = bt
                    frag.RemoveBond(atom1,nbr_idx)
        
        
        # mol.RemoveBond(mol_iface[0],mol_iface[2])
        # mol.RemoveBond(mol_iface[1],mol_iface[3])
        frag.RemoveBond(frag_iface[0],frag_iface[1])
        self.atom_map[edge[1]] = []

        for atom in frag.GetAtoms():
            at_idx = atom.GetIdx()
            # atom.SetFormalCharge(0)
            new_at_idx = mol.AddAtom(atom)
            
            self.atom_map[edge[1]].append(new_at_idx)
            x,y,z = f_conf.GetAtomPosition(at_idx)
            conf.SetAtomPosition(new_at_idx,Point3D(x,y,z))
        for bond in frag.GetBonds():
            mol.AddBond(self.atom_map[edge[1]][bond.GetBeginAtomIdx()],
                        self.atom_map[edge[1]][bond.GetEndAtomIdx()],
                        order= bond.GetBondType())

        
        frag_iface = [self.atom_map[edge[1]][at] for at in frag_iface]
        x1,y1,z1 = conf.GetAtomPosition(mol_iface[2])
        x2,y2,z2 = conf.GetAtomPosition(mol_iface[3])
        x3,y3,z3 = conf.GetAtomPosition(frag_iface[2])
        x4,y4,z4 = conf.GetAtomPosition(frag_iface[3])
        dummy1= mol.AddAtom(Chem.Atom(80))
        dummy2= mol.AddAtom(Chem.Atom(80))
        # print((x1+x4)/2,(y3+y4)/2,(z3+z4)/2)
        conf.SetAtomPosition(dummy1,Point3D((x1+x2)/2,(y1+y2)/2,(z1+z2)/2))
        conf.SetAtomPosition(dummy2,Point3D((x3+x4)/2,(y3+y4)/2,(z3+z4)/2))

        

        mol.AddBond(mol_iface[0],dummy1)
        mol.AddBond(mol_iface[1],dummy2)
        mol.AddBond(frag_iface[3],dummy2)
        mol.AddBond(mol_iface[2],dummy1)
        Chem.SanitizeMol(mol,sanitizeOps=Chem.SanitizeFlags.SANITIZE_SYMMRINGS)
        rT.SetDihedralDeg(conf,dummy1,mol_iface[0],mol_iface[1],dummy2,-dihedral)
        # rT.SetDihedralDeg(conf,dummy1,mol_iface[0],mol_iface[1],dummy2,-dihedral)
        

        mol.RemoveBond(mol_iface[0],dummy1)
        mol.RemoveBond(mol_iface[1],dummy2)
        mol.RemoveBond(frag_iface[3],dummy2)
        mol.RemoveBond(mol_iface[2],dummy1)
        # atom ,nbr atom in 
        for bond in frag_remove_bonds:
            bt = frag_remove_bonds[bond]
            atom,nbr =  bond
            nbr = self.atom_map[edge[1]][nbr]
            if atom == ori_frag_atom0:
                mol.AddBond(mol_iface[0],nbr,order=bt)
            else:
                mol.AddBond(mol_iface[1],nbr,order=bt)
        for bond in mol_remove_bonds:
            bt = mol_remove_bonds[bond]
            atom,nbr =  bond
            mol.AddBond(atom,nbr,order=bt)
        mol.GetAtomWithIdx(frag_iface[0]).SetIsAromatic(0)
        mol.GetAtomWithIdx(frag_iface[1]).SetIsAromatic(0)
        self.atom_map[edge[1]][ori_frag_atom0] = mol_iface[0]
        self.atom_map[edge[1]][ori_frag_atom1] = mol_iface[1]
        # mol.GetAtomWithIdx(mol_iface[0]).SetNumExplicitHs(max(mol.GetAtomWithIdx(mol_iface[0]).GetNumExplicitHs()-1,0))
        # mol.GetAtomWithIdx(mol_iface[1]).SetNumExplicitHs(max(mol.GetAtomWithIdx(mol_iface[1]).GetNumExplicitHs()-1,0))
        # atom = mol.GetAtomWithIdx(mol_iface[0])
        mol.GetAtomWithIdx(mol_iface[0]).SetNumExplicitHs(0)
        mol.GetAtomWithIdx(mol_iface[1]).SetNumExplicitHs(0)
    
    def asm_plain_pah(self,mol,edge,frag1_iface,frag2_iface,reflection = False):
        # conf1_index = self.d3_index[edge[0]]
        # conf2_index = self.d3_index[edge[1]]
        # iface1_index,iface2_index = self.iface_index[edge]
        conf = mol.GetConformer(0)
        if reflection:
            frag2_iface = swap_atom_order(frag2_iface)

        if frag1_iface[0] == frag2_iface[0]:
            common = frag1_iface[0]
            frag1_atom,frag2_atom = frag1_iface[1],frag2_iface[1]
            # frag2_atoms = [frag2_iface[1],frag2_iface[3]]
        else:
            common = frag1_iface[1]
            frag1_atom,frag2_atom = frag1_iface[0],frag2_iface[0]
            # frag1_atom = [frag1_iface[0],frag1_iface[2]]
            # frag2_atoms = [frag2_iface[0],frag2_iface[2]]
        # print(common,frag1_atoms,frag2_atoms)

        bond_type1 = mol.GetBondBetweenAtoms(common,frag1_atom).GetBondType()
        bond_type2 = mol.GetBondBetweenAtoms(common,frag2_atom).GetBondType()
        if Chem.BondType.AROMATIC in (bond_type1,bond_type2):
            commom_bond_type = Chem.BondType.AROMATIC
        else:
            commom_bond_type = bond_type1

        mol.RemoveBond(common,frag1_atom)
        mol.RemoveBond(common,frag2_atom)
        remove_nbr_bonds = {}

        for at_idx in [frag1_atom,frag2_atom]:
            for nbr in mol.GetAtomWithIdx(at_idx).GetNeighbors():
                nbr_idx = nbr.GetIdx()
                if nbr_idx != common:
                    remove_nbr_bonds[nbr_idx] = mol.GetBondBetweenAtoms(at_idx,nbr_idx).GetBondType()
                    mol.RemoveBond(at_idx,nbr_idx)
        
        atom = mol.GetAtomWithIdx(frag1_atom)
        new_atom = Chem.Atom(atom.GetSymbol())
        new_at_idx = mol.AddAtom(new_atom)
        x1,y1,z1 = conf.GetAtomPosition(frag1_atom)
        x2,y2,z2 = conf.GetAtomPosition(frag2_atom)
        conf.SetAtomPosition(new_at_idx,Point3D((x1+x2)/2,(y1+y2)/2,(z1+z2)/2))
        mol.GetAtomWithIdx(frag1_atom).SetIsAromatic(False)
        mol.GetAtomWithIdx(frag2_atom).SetIsAromatic(False)

        self.atom_map[edge[0]] = [new_at_idx if at == frag1_atom else at  for at in self.atom_map[edge[0]]]
        self.atom_map[edge[1]] = [new_at_idx if at == frag2_atom else at  for at in self.atom_map[edge[1]]]
        # atom_map[edge[0]][frag1_atom] = new_at_idx
        # atom_map[edge[1]][frag2_atom] = new_at_idx
        mol.AddBond(new_at_idx,common,order= commom_bond_type)
        for nbr_idx in remove_nbr_bonds:
            mol.AddBond(new_at_idx,nbr_idx,order= remove_nbr_bonds[nbr_idx])
        mol.GetAtomWithIdx(new_at_idx).SetNumExplicitHs(0)  
        # mol.GetAtomWithIdx(new_at_idx).SetNumExplicitHs(max(mol.GetAtomWithIdx(new_at_idx).GetNumExplicitHs()-1,0))

        

    def angle2idx(self,a):
        a+= 180+(self.angle_interval/2)
        return int((a%360)/self.angle_interval)

    def idx2angle(self,idx):
        return idx*self.angle_interval-180


if __name__ == "__main__":

    from segmentor.frag_seg import FragSeg
    from data.vocabulary import FragVocab

    from rdkit.Chem.rdMolAlign import AlignMol
    from tqdm import tqdm
    import math
    from rdkit import Chem

    import pickle
    mols = pickle.load(open('geom-drugs/test.mol','rb'))
    hit_vocab = pickle.load(open('geom-drugs/hit.pkl','rb'))
    vocab_file = pickle.load(open('geom-drugs/vocab.pkl','rb'))
    seg = FragSeg(hit_vocab=hit_vocab)
    vocab = FragVocab(vocab_file=vocab_file)
    processor  = FragProcessor(segmentor= seg, vocab =vocab )

    rmsd = []
    count=-1
    for mol in tqdm(mols):
        count+=1
        info = processor.fragmentize_sanitized_mol(mol)
        try:
        
            new_mol = processor.assemble_mol(conf_index=info['conf_index'], 
                                                dihedrals = info['dihedrals'], 
                                                edge_index=info['edge_index'],
                                                iface_index=info['iface_index'], 
                                                iface_types= info['iface_types'])
        except:
            print("ASM ERROR",Chem.MolToSmiles(mol),count)
            continue
        if not new_mol:
            print("ASM2 ERROR",Chem.MolToSmiles(mol),count)
            continue
        try:
            r = AlignMol(mol,new_mol)
        except Exception as e:
            print("ALIGN ERROR",Chem.MolToSmiles(mol),Chem.MolToSmiles(new_mol),count)
            continue
        if not math.isnan(r):
            rmsd.append(r)
    print(sum(rmsd)/len(rmsd),len(rmsd))

        