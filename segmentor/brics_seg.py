from rdkit import Chem
from rdkit.Chem.BRICS import FindBRICSBonds

class Single_Bond_Seg:
    def fragmentize(self, mol, dummyStart=1):
        bond_ids = [b.GetIdx() for b in mol.GetBonds() if (b.GetBondType() is Chem.BondType.SINGLE) and (not b.IsInRing())]
        if len(bond_ids) != 0:
            # break the bonds & set the dummy labels for the bonds
            dummyLabels = [(i + dummyStart, i + dummyStart) for i in range(len(bond_ids))]
            break_mol = Chem.FragmentOnBonds(mol, bond_ids, dummyLabels=dummyLabels)
            dummyEnd = dummyStart + len(dummyLabels) - 1
        else:
            break_mol = mol
            dummyEnd = dummyStart - 1
        return list(Chem.rdmolops.GetMolFrags(break_mol, asMols=True)), dummyEnd


class BRICS_Seg():
    def __inti__(self):
        self.type = 'BRICS_Fragmenizers'
    
    def get_bonds(self, mol):
        bonds = [bond[0] for bond in list(FindBRICSBonds(mol))]
        return bonds
    
    def fragmentize(self, mol, dummyStart=1):
        # get bonds need to be break
        bonds = [bond[0] for bond in list(FindBRICSBonds(mol))]
        
        # whether the molecule can really be break
        if len(bonds) != 0:
            bond_ids = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bonds]

            # break the bonds & set the dummy labels for the bonds
            dummyLabels = [(i + dummyStart, i + dummyStart) for i in range(len(bond_ids))]
            break_mol = Chem.FragmentOnBonds(mol, bond_ids, dummyLabels=dummyLabels)
            dummyEnd = dummyStart + len(dummyLabels) - 1
        else:
            break_mol = mol
            dummyEnd = dummyStart - 1

        return list(Chem.rdmolops.GetMolFrags(break_mol, asMols=True)), dummyEnd

class RING_R_Seg():

    def bonds_filter(self, mol, bonds):
        filted_bonds = []
        bonds = self.search_other_bonds(mol,bonds)
        for bond in bonds:
            bond =self._filter(mol,bond)
            if bond:
                filted_bonds.append(bond)
        return filted_bonds

    def _filter(self,mol,bond):
        f_atom = mol.GetAtomWithIdx(bond[0])
        s_atom = mol.GetAtomWithIdx(bond[1])
        if f_atom.GetSymbol() == '*' or s_atom.GetSymbol() == '*':
            return None
        if mol.GetBondBetweenAtoms(bond[0], bond[1]).IsInRing():
            return None
        return bond

    def get_bonds(self, mol):
        bonds = []
        rings = self.get_rings(mol)
        if len(rings) > 0:
            for ring in rings:
                rest_atom_idx = self.get_other_atom_idx(mol, ring)
                bonds += self.find_parts_bonds(mol, [rest_atom_idx, ring])
            bonds = self.bonds_filter(mol, bonds)
        return bonds
    
    def fragmentize(self, mol, dummyStart=1):
        rings = self.get_rings(mol)
        if len(rings) > 0:
            bonds = []
            for ring in rings:
                rest_atom_idx = self.get_other_atom_idx(mol, ring)
                bonds += self.find_parts_bonds(mol, [rest_atom_idx, ring])
            bonds = self.bonds_filter(mol, bonds)
            if len(bonds) > 0:
                bond_ids = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bonds]
                bond_ids = list(set(bond_ids))
                dummyLabels = [(i + dummyStart, i + dummyStart) for i in range(len(bond_ids))]
                break_mol = Chem.FragmentOnBonds(mol, bond_ids, dummyLabels=dummyLabels)
                dummyEnd = dummyStart + len(dummyLabels) - 1
            else:
                break_mol = mol
                dummyEnd = dummyStart - 1
        else:
            break_mol = mol
            dummyEnd = dummyStart - 1
        return list(Chem.rdmolops.GetMolFrags(break_mol, asMols=True)), dummyEnd

    def search_other_bonds(self,mol,bonds):
        new_bonds=[]
        for bond in bonds:
            bond_type = mol.GetBondBetweenAtoms(bond[0], bond[1]).GetBondType()
            if bond_type is not Chem.BondType.SINGLE:
                far_ring_atom=mol.GetAtomWithIdx(bond[0])
                far_ring_atom_nbr= far_ring_atom.GetNeighbors()
                for atom in far_ring_atom_nbr:
                    if atom.GetIdx()!=bond[1]:
                        new_bond = mol.GetBondBetweenAtoms(bond[0], atom.GetIdx())
                        if new_bond.GetBondType() is Chem.BondType.SINGLE and not new_bond.IsInRing():
                            new_bonds.append([bond[0],atom.GetIdx()])
            else:
                new_bonds.append(bond)
        return new_bonds

    def find_parts_bonds(self,mol, parts):
        ret_bonds = []
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                i_part = parts[i]
                j_part = parts[j]
                for i_atom_idx in i_part:
                    for j_atom_idx in j_part:
                        bond = mol.GetBondBetweenAtoms(i_atom_idx, j_atom_idx)
                        if bond is None:
                            continue
                        ret_bonds.append((i_atom_idx, j_atom_idx))
        return ret_bonds

    def get_other_atom_idx(self,mol, atom_idx_list):
        ret_atom_idx = []
        for atom in mol.GetAtoms():
            if atom.GetIdx() not in atom_idx_list:
                ret_atom_idx.append(atom.GetIdx())
        return ret_atom_idx

    def get_rings(self,mol):
        rings = []
        for ring in list(Chem.GetSymmSSSR(mol)):
            ring = list(ring)
            rings.append(ring)
        return rings


class BRICS_RING_R_Seg():
    def __init__(self):
        self.brics_seg = BRICS_Seg()
        self.ring_r_seg = RING_R_Seg()
    
    def fragmentize(self, mol, dummyStart=1):
        brics_bonds = self.brics_seg.get_bonds(mol)
        ring_r_bonds = self.ring_r_seg.get_bonds(mol)
        bonds = brics_bonds + ring_r_bonds
        bond_ids=[]
        for x, y in bonds:
            bond = mol.GetBondBetweenAtoms(x, y)
            if bond.GetBondType() is Chem.BondType.SINGLE or bond.GetBondType() is Chem.BondType.AROMATIC:
                bond_ids.append(bond.GetIdx())
        if len(bond_ids) != 0:
            bond_ids = list(set(bond_ids))

            dummyLabels = [(i + dummyStart, i + dummyStart) for i in range(len(bond_ids))]
            break_mol = Chem.FragmentOnBonds(mol, bond_ids, dummyLabels=dummyLabels)
            dummyEnd = dummyStart + len(dummyLabels)
        else:
            break_mol = mol
            dummyEnd = dummyStart

        return list(Chem.rdmolops.GetMolFrags(break_mol, asMols=True)), dummyEnd


