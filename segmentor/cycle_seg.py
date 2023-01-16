import rdkit.Chem as Chem
from copy import deepcopy
from rdkit.Geometry import Point3D
import networkx as nx

def GetBondNeighbors(mol,bond_idx):
    nbr_bonds = []
    bond = mol.GetBondWithIdx(bond_idx)
    for at_idx,oat_idx in [(bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()),(bond.GetEndAtomIdx(),bond.GetBeginAtomIdx())]:
        for nbr in mol.GetAtomWithIdx(at_idx).GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx != oat_idx:
                nbr_bonds.append(mol.GetBondBetweenAtoms(at_idx,nbr_idx).GetIdx())
    return nbr_bonds

def GetIsAromatic(mol,bonds):
    isAromatic = True
    for bond_idx in bonds:
        bond = mol.GetBondWithIdx(bond_idx)
        isAromatic &= bond.GetBeginAtom().GetIsAromatic()
        isAromatic &= bond.GetEndAtom().GetIsAromatic()
    return isAromatic
    
def GetDoubleBondNum(mol,bonds):
    double = 0
    for bond_idx in bonds:
        if mol.GetBondWithIdx(bond_idx).GetBondType() is Chem.BondType.DOUBLE:
            double +=1
    return double
    
def GetBondMap(mol):
    bond_map = {}
    for bond in mol.GetBonds():
        bat,eat = bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        bond_map[bond.GetIdx()] = (bond.GetBeginAtomIdx(),bond.GetEndAtomIdx())
    return bond_map

def GetAtomSet(bonds,bond_map):
    atoms = set()
    for bond in bonds:
        atoms |= set(bond_map[bond])
    return list(atoms)

class CycleSeg:

    def fragmentize(self,mol,dummyStart=1):
        self.dummyEnd = dummyStart
        # self.atom_label_idx = dummyStart 
        self.overlap = {}
        frags = []
        # rw_mol = Chem.RWMol(deepcopy(mol))
        cpts = self.split_plain_cycles(Chem.RWMol(deepcopy(mol)))
        for frag in cpts:
            frags+=self.split_spiro_cycles(Chem.RWMol(deepcopy(frag)))

        return frags, self.dummyEnd

    def get_spiro_cycles(self,mol):
        ri =mol.GetRingInfo()
        cycles = [set(x) for x in ri.AtomRings()]
        if len(cycles)==1:
            return cycles
        def get_component(cycles):
            F  = nx.Graph()
            F.add_nodes_from(list(range(len(cycles))))
            overlap  = set()
            for i in range(len(cycles)-1):
                for j in range(i+1,len(cycles)):
                    o = list(cycles[i] & cycles[j])
                    if len(o)>1:
                        F.add_edge(i,j)
            cpt = [x for x in nx.connected_components(F)]
            return cpt
        components = get_component(cycles)
        while len(cycles) > len(components):
            cycles = [set().union(*[cycles[i] for i in cpt]) for cpt in components]
            components = get_component(cycles)
        return [set(c) for c in cycles]
        

    def split_spiro_cycles(self,mol):
        spiros ={}
        cycles = self.get_spiro_cycles(mol)
        for i in range(len(cycles)-1):
            for j in range(i+1,len(cycles)):
                o = list(cycles[i] & cycles[j])
                if len(o)==1 and o[0] not in spiros:
                    spiros[o[0]] = []
                    for at in mol.GetAtomWithIdx(o[0]).GetNeighbors():
                        at_idx = at.GetIdx()
                        if at_idx not in cycles[i]:
                            spiros[o[0]].append(at_idx)
        
        if spiros:
            atom_map = {}
            conf = mol.GetConformers()
            conf = conf[0] if len(conf) >0 else None
            for atom_idx in spiros:
                self.dummyEnd +=1
                # atom_idx = spiro[0]
                atom = mol.GetAtomWithIdx(atom_idx)
                atom.SetIsotope(self.dummyEnd)
                new_atom = Chem.Atom(atom.GetSymbol())
                new_atom.SetIsotope(self.dummyEnd)
                new_idx = mol.AddAtom(new_atom)
                atom_map[atom_idx] = new_idx
                # print(spiros)
            bond_set = set()
            for atom_idx in spiros:
                for nbr_idx in spiros[atom_idx]:
                    if (atom_idx,nbr_idx) in bond_set or (nbr_idx,atom_idx) in bond_set:
                        continue
                    else:
                        bond_set.add(((atom_idx,nbr_idx)))
                    new_atom_idx = atom_map[atom_idx]
                    new_nbr_idx = atom_map[nbr_idx] if nbr_idx in atom_map else nbr_idx
                    bond = mol.GetBondBetweenAtoms(atom_idx,nbr_idx)
                    bt = bond.GetBondType()
                    bond_tag = bond.GetProp("idx") if bond.HasProp("idx") else 0
                    mol.RemoveBond(atom_idx,nbr_idx)
                    mol.AddBond(new_atom_idx,new_nbr_idx,order = bt)
                    if bond_tag:
                        mol.GetBondBetweenAtoms(new_atom_idx,new_nbr_idx).SetIntProp("idx",int(bond_tag))
                if conf:
                    x,y,z = conf.GetAtomPosition(atom_idx)
                    conf.SetAtomPosition(new_idx,Point3D(x,y,z))
        mol.UpdatePropertyCache()
        Chem.SanitizeMol(mol)
        # list(Chem.rdmolops.GetMolFrags(mol, asMols=True))   
        return list(Chem.rdmolops.GetMolFrags(mol, asMols=True))
        
                    
    def get_plain_cycles(self,mol):
        ri =mol.GetRingInfo()
        cycles = [set(x) for x in ri.BondRings()]
        if len(cycles)==1:
            return cycles
        def get_component(cycles):
            F  = nx.Graph()
            F.add_nodes_from(list(range(len(cycles))))
            overlap  = set()
            for i in range(len(cycles)-1):
                for j in range(i+1,len(cycles)):
                    o = list(cycles[i] & cycles[j])
                    if o:
                        if len(o)>1 or o[0] in overlap:
                            F.add_edge(i,j)
                        else:
                            overlap.add(o[0])
            cpt = [x for x in nx.connected_components(F)]
            return cpt
        components = get_component(cycles)
        while len(cycles) > len(components):
            cycles = [set().union(*[cycles[i] for i in cpt]) for cpt in components]
            components = get_component(cycles)
        return [list(c) for c in cycles]

    def search_cycle_to_split(self,mol,cycles,bond_map):
        aromatics = [GetIsAromatic(mol,cycle) for cycle in cycles]
        cycles_atoms = [GetAtomSet(cycle,bond_map) for cycle in cycles]
        for i in range(len(cycles)):
                i_bonds = {}
                j_bonds = {}
                i_bonds_set =set()
                j_bonds_set =set()
                overlap = set()
                spiro_atoms = set()         
                for j in range(len(cycles)):
                    if i !=j:
                        o = tuple(set(cycles[i]) & set(cycles[j]))
                        if o:
                            i_bonds[j] = set()
                            j_bonds[j] = set()
                            overlap.add(o[0])
                            nbr_bonds = GetBondNeighbors(mol,o[0])
                            for nbr_bond in nbr_bonds:
                                i_bonds[j].add(nbr_bond) if  nbr_bond in cycles[i] else j_bonds[j].add(nbr_bond)
                                i_bonds_set.add(nbr_bond) if  nbr_bond in cycles[i] else j_bonds_set.add(nbr_bond)
                        else:
                            o = tuple(set(cycles_atoms[i]) & set(cycles_atoms[j]))
                            if len(o) >0:
                                spiro_atoms.add(o[0]) 

                not_aromatics = True
                for j in i_bonds:
                    not_aromatics&= (not aromatics[j])
                i_bonds_set -= overlap
                j_bonds_set -= overlap
                if GetDoubleBondNum(mol, i_bonds_set)%2 != 0 and not not_aromatics:
                    continue
                if GetDoubleBondNum(mol, j_bonds_set)%2 != 0 and aromatics[i]:
                    continue
                
                flag = True
                for j in i_bonds:
                    if GetDoubleBondNum(mol, i_bonds[j])%2 != 0 and aromatics[j]:
                        flag = False
                for j in j_bonds:
                    if GetDoubleBondNum(mol, j_bonds[j])%2 != 0 and aromatics[i]:
                        flag = False
                
                if flag:
                    extra_info = {"ihs":{},"jhs":{},'spiro':spiro_atoms}
                    i_bonds = {j:list(bonds-overlap) for j,bonds in i_bonds.items()}  
                    j_bonds = {j:list(bonds-overlap) for j,bonds in j_bonds.items()}
                    bond_atom_set = set()
                    for j in j_bonds:
                        for bond_idx in j_bonds[j]:
                            cycle_atoms = cycles_atoms[i]
                            bond = mol.GetBondWithIdx(bond_idx)
                            bat,eat = bond_map[bond_idx]
                            at_idx = bat if bat in cycle_atoms else eat
                            bt = bond.GetBondType()
                            
                            if (bond_idx,at_idx) not in bond_atom_set:
                                bond_atom_set.add((bond_idx,at_idx))
                            else:
                                continue
                            if at_idx not in extra_info['jhs']:
                                extra_info['jhs'][at_idx] = 0
                            if aromatics[i]:
                                extra_info['jhs'][at_idx] = 1 
                            elif bt is Chem.BondType.DOUBLE:
                                extra_info['jhs'][at_idx] += 2 
                            else:
                                extra_info['jhs'][at_idx] += 1
                                
                    bond_atom_set = set()     
                    for j in i_bonds:
                        for bond_idx in i_bonds[j]:
                            cycle_atoms = cycles_atoms[j]
                            bond = mol.GetBondWithIdx(bond_idx)
                            bat,eat = bond_map[bond_idx]
                            at_idx = bat if bat in cycle_atoms else eat
                            bt = bond.GetBondType()
                            if (bond_idx,at_idx) not in bond_atom_set:
                                bond_atom_set.add((bond_idx,at_idx))
                            else:
                                continue
                            if at_idx not in extra_info['ihs']:
                                extra_info['ihs'][at_idx] = 0
                            if aromatics[j]:
                                extra_info['ihs'][at_idx] = 1 
                            elif bt is Chem.BondType.DOUBLE:
                                extra_info['ihs'][at_idx] += 2  
                            else:
                                extra_info['ihs'][at_idx] += 1
                    return i,i_bonds_set,overlap,extra_info
        return -1,None,None,None
    
    def split_plain_cycles(self,mol): 
        frags = []
        bond_map = {}
        conf = mol.GetConformers()
        conf = conf[0] if len(conf) >0 else None
        
        cycles = self.get_plain_cycles(mol)
        bond_map = GetBondMap(mol)
        kekulized_mol = deepcopy(mol)
        Chem.Kekulize(kekulized_mol)
        cycle_id,bonds,overlap,extra_info  = self.search_cycle_to_split(kekulized_mol,cycles,bond_map)
        while cycle_id>=0 and len(cycles)>1:
            
            cycle = cycles[cycle_id]
            isAromatic = GetIsAromatic(mol,cycle)
            cycles.pop(cycle_id)
            atom_map = {}
            for bond_idx in overlap:
                self.dummyEnd +=1
                atoms = bond_map[bond_idx]
                bond = mol.GetBondBetweenAtoms(*atoms)
                bond_id = self.dummyEnd
                bond.SetIntProp("idx",bond_id)
                # print(self.dummyEnd,atoms)
                bt = kekulized_mol.GetBondBetweenAtoms(*atoms).GetBondType() 
                
                new_atoms = []
                for at_idx in atoms:
                    if at_idx not in atom_map:
                        atom  = mol.GetAtomWithIdx(at_idx)
                        if not atom.GetAtomMapNum():
                            self.dummyEnd +=1
                            atom.SetAtomMapNum(self.dummyEnd)
                        new_atom = Chem.Atom(atom.GetSymbol())
                        new_atom.SetFormalCharge(atom.GetFormalCharge())
                        new_atom.SetNumExplicitHs(extra_info['jhs'][at_idx])
                        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
                        # new_atom.SetIsAromatic(False) if not isAromatic and atom.GetIsAromatic() else new_atom.SetIsAromatic(atom.GetIsAromatic())
                        new_idx = mol.AddAtom(new_atom)
                        if conf:
                            x,y,z = conf.GetAtomPosition(at_idx)
                            conf.SetAtomPosition(new_idx,Point3D(x,y,z))
                        new_atoms.append(new_idx)
                        atom_map[at_idx] = new_idx
                    else:
                        new_atoms.append(atom_map[at_idx])
                mol.AddBond(new_atoms[0],new_atoms[1],order=bt if not isAromatic else Chem.BondType.AROMATIC)
                mol.GetBondBetweenAtoms(new_atoms[0],new_atoms[1]).SetIntProp("idx",bond_id)
                # print(self.dummyEnd,'new',new_atoms[0],new_atoms[1],atom_map)
            for bond_idx in bonds:
                bat,eat = bond_map[bond_idx]
                new_bat = atom_map[bat] if bat in atom_map else bat
                new_eat = atom_map[eat] if eat in atom_map else eat
                bond = kekulized_mol.GetBondBetweenAtoms(bat,eat)
                bond_tag = bond.GetProp("idx") if bond.HasProp("idx") else 0
                bt = bond.GetBondType() 
                mol.RemoveBond(bat,eat)
                mol.AddBond(new_bat,new_eat,order=bt if not isAromatic else Chem.BondType.AROMATIC)
                if bond_tag:
                    mol.GetBondBetweenAtoms(new_bat,new_eat).SetIntProp("idx",int(bond_tag))
            for at_idx in extra_info['ihs']:
                mol.GetAtomWithIdx(at_idx).SetNumExplicitHs(extra_info['ihs'][at_idx])
            
            for at_idx in extra_info['spiro']:
                self.dummyEnd+=1
                mol.GetAtomWithIdx(at_idx).SetIsotope(self.dummyEnd)
                if at_idx in atom_map:
                    mol.GetAtomWithIdx(atom_map[at_idx]).SetIsotope(self.dummyEnd)

            mol.UpdatePropertyCache()
            Chem.SanitizeMol(mol)
            for cycle in cycles:
                for idx in range(len(cycle)):
                    atoms = bond_map[cycle[idx]]
                    cycle[idx] = mol.GetBondBetweenAtoms(*atoms).GetIdx()
            bond_map = GetBondMap(mol)
            kekulized_mol = deepcopy(mol)
            Chem.Kekulize(kekulized_mol)
            cycle_id,bonds,overlap,extra_info  = self.search_cycle_to_split(kekulized_mol,cycles,bond_map)
        # for bond in mol.GetBonds():
        #     if bond.HasProp("idx"):
        #         print(bond.GetBeginAtomIdx(),bond.GetEndAtomIdx(),bond.GetProp("idx"))
        return list(Chem.rdmolops.GetMolFrags(mol, asMols=True))




