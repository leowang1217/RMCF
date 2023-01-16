from utils.fragment import canonical_frag_smi,get_clean_smi,reorder_frag,wash_frag
from rdkit import Chem
from copy import deepcopy
from tqdm import tqdm
import pickle

class  ChainSeg():
    def __init__(self,hit_vocab = None):
        if hit_vocab:
            self.hit_vocab=hit_vocab
        else:
            self.hit_vocab={}
        self.rigid_frags = self.get_rigid_frags()

    def get_rigid_frags(self):
        vocab=[]
        for smi in self.hit_vocab:
            if len(self.hit_vocab[smi]) ==1:
                vocab.append((Chem.MolFromSmarts(smi),smi,sum([x[1] for x in self.hit_vocab[smi]])))
        vocab.sort(key=lambda x:len(x[0].GetAtoms()),reverse=True)
        return vocab

    def is_valid(self,frags):
        def if_split_single(frag):
            for atom in frag.GetAtoms():
                if atom.GetSymbol()=='*':
                    bonds=atom.GetBonds()
                    if bonds[0].GetBondType() is not Chem.BondType.SINGLE:
                        return False            
            return True
        for frag in frags:
            frag_smi = canonical_frag_smi(Chem.MolToSmiles(frag))
            if (not if_split_single(frag)) or (frag_smi=='**') or ('*' not in frag_smi):
                return False
        return True

    def has_substruct(self,mol,p):
        match = mol.GetSubstructMatch(p,useChirality=True)
        if not match:
            return False
        else:
            p_atoms=p.GetAtoms()
            mol_atoms = mol.GetAtoms()
            for i,j in enumerate(match):
                m_atom = mol_atoms[j]
                bonds = m_atom.GetBonds()
                for b in bonds:
                    bat,eat = b.GetBeginAtomIdx(),b.GetEndAtomIdx()
                    if (bat in match ) ^ (eat in match) and p_atoms[i].GetSymbol()!='*':
                        return False
            return True
    def remove_core(self,mol,p):
        # mol = deepcopy(mol)
        for at in mol.GetAtoms():
            at.SetIsotope(0)
        match = mol.GetSubstructMatch(p,useChirality=True)
        # print(match)
        remove_bonds = set()
        remove_atoms = [j for i,j in enumerate(match) if p.GetAtomWithIdx(i).GetSymbol()!='*' or mol.GetAtomWithIdx(j).GetSymbol()=='*']
        for at_idx in remove_atoms:
            for bond in mol.GetAtomWithIdx(at_idx).GetBonds():
                    if (bond.GetEndAtomIdx() not in remove_atoms) or  (bond.GetBeginAtomIdx() not in remove_atoms):
                        remove_bonds.add(bond.GetIdx())
        if not remove_bonds:
            return [],[]
        remove_bond_atom=[(mol.GetBondWithIdx(b).GetBeginAtom().GetIdx(),mol.GetBondWithIdx(b).GetEndAtom().GetIdx(),) for b in remove_bonds]
    
        dummyLabels = [(0, 0) for i in range(len(remove_bonds))]
        for idx in remove_atoms:
            mol.GetAtomWithIdx(idx).SetIsotope(2022)
        frags = Chem.FragmentOnBonds(mol,list(remove_bonds),dummyLabels=dummyLabels)
        
        frags = list(Chem.rdmolops.GetMolFrags(frags, asMols=True))
        # print(get_clean_smi(mol),get_clean_smi(p),[(get_clean_smi(f),Chem.MolToSmiles(f)) for f in frags],Chem.MolToSmiles(mol))
        target_idx= None

        for i in range(len(frags)):
            for atom in frags[i].GetAtoms():
                if atom.GetSymbol()=='*':
                    continue
                else:
                    if atom.GetIsotope() ==2022:
                        target_idx=i
                    break
            if target_idx is not None:
                break
        if target_idx is not None: 
            frags.pop(target_idx)
        return frags,remove_bond_atom  

    def dp_search(self,smiles,gready_search=False,use_hit =False):
        if use_hit and (smiles in self.hit_vocab):
            return self.hit_vocab[smiles]
        # else:
        #     print('not hit',smiles)
        mol = Chem.MolFromSmiles(smiles)
        # for at in mol.GetAtoms():
        #     at.SetAtomMapNum(at.GetIdx())
        def dp(mol):
            mol = deepcopy(mol)
            
            smi = get_clean_smi(mol)
            if use_hit and (smi in self.hit_vocab):
                return self.hit_vocab[smi]
            for at in mol.GetAtoms():
                # at.SetIsotope(0)
                at.SetAtomMapNum(at.GetIdx())
            
            best_score = float('inf')
            best_split=[(Chem.MolToSmiles(mol),10000,[])]
            for p,p_smi,score in self.rigid_frags:
                flag = self.has_substruct(mol,p)
                if flag:
                    frags,remove_bond = self.remove_core(mol,p)
                    valid  = self.is_valid(frags)
                    if valid:
                        cur_score = score
                        cur_split =[(p_smi,score,remove_bond)]
                        if not frags and (p_smi!= smi):
                            continue
                        
                        # print(smi,Chem.MolToSmiles(mol),p_smi,Chem.MolToSmiles(p),[get_clean_smi(f) for f in frags])
                        for f in frags:
                            f = reorder_frag(f)
                            frag_best_split = dp(f)
                            # print(Chem.MolToSmiles(f))
                            frag_best_split = [(x[0],x[1],[(f.GetAtomWithIdx(y).GetAtomMapNum(),f.GetAtomWithIdx(z).GetAtomMapNum()) for y,z in x[2]]) for x in frag_best_split]
                            
                            cur_split+=frag_best_split
                            cur_score+= sum([x[1] for x in frag_best_split])
                        
                        if cur_score < best_score:
                            best_split = cur_split
                            best_score = cur_score
                        if gready_search:
                            break
            if use_hit and (smi not in self.hit_vocab):
                self.hit_vocab[smi] = best_split
            return best_split
        return dp(mol)

    def fragmentize(self,mol,dummyStart=1):
        # smi = get_clean_smi(mol)
        mol = reorder_frag(mol)
        smi = get_clean_smi(mol)
        best_split = self.dp_search(smi,gready_search=True,use_hit=True)
        # print(smi,best_split)
        bonds = [mol.GetBondBetweenAtoms(x[0],x[1]).GetIdx() for s in best_split for x in s[2] if x]
        dummyLabels = [(i + dummyStart, i + dummyStart) for i in range(len(bonds))]
        dummyEnd = dummyStart+ len(bonds)
        if bonds:
            frags = Chem.FragmentOnBonds(mol,list(bonds),dummyLabels=dummyLabels)
            return list(Chem.rdmolops.GetMolFrags(frags, asMols=True)),dummyEnd
        else:
            return [mol],dummyStart

    def gen_vocab(self,chain_score):
        self.rigid_frags=[]
        for smi in chain_score:
            self.rigid_frags.append((Chem.MolFromSmarts(smi),smi,chain_score[smi]))
        self.rigid_frags.sort(key=lambda x:len(x[0].GetAtoms()),reverse=True)
        smis = [get_clean_smi(Chem.MolFromSmiles(smi)) for smi in chain_score]
        smis.sort(key = lambda x:len(Chem.MolFromSmiles(x).GetAtoms()))
        for smi in tqdm(smis):
            # try:
            self.dp_search(smi,gready_search=False,use_hit=True)
        return self.hit_vocab
        