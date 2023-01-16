from segmentor.brics_seg import BRICS_RING_R_Seg
from segmentor.chain_seg import ChainSeg
from segmentor.cycle_seg import CycleSeg
from rdkit import Chem
from utils.fragment import reorder_frag


def get_plain_iface_type(atom_pairs1,atom_pairs2):
    a1 = atom_pairs1[0]
    a2 = atom_pairs1[1]
    b1 = atom_pairs2[0]
    b2 = atom_pairs2[1]
    # print(a1,a2,b1,b2)
    return not ((a1<a2)^(b1<b2))
    
def sort_atoms(atoms):
    if len(atoms)==2 and atoms[0] > atoms[1]:
            return (atoms[1],atoms[0])
    else:
        return atoms
     
class FragSeg:
    def __init__(self,hit_vocab):
        self.brics_ring_seg = BRICS_RING_R_Seg()
        self.chain_sg = ChainSeg(hit_vocab=hit_vocab)
        self.cycle_sg = CycleSeg()
    

        
    def fragmenize(self, mol, dummyStart=1,return_graph=False):
        frags,dummyEnd  = self.brics_ring_seg.fragmentize(mol,dummyStart=dummyStart)
        end = dummyEnd
        final_frags=[]
        # final_map ={}
        for frag in frags:
            num_rings = len(Chem.GetSSSR(frag))
            # print(num_rings)
            if num_rings==0:
                # print(Chem.MolToSmiles(frag))
                add_frags,end = self.chain_sg.fragmentize(frag,dummyStart=end)
            else:
                add_frags,end = self.cycle_sg.fragmentize(frag,dummyStart=end)
            for mol in add_frags:
                final_frags.append(reorder_frag(mol))
        if return_graph:
            graph = self.get_graph(final_frags)
            return  final_frags,end,graph
        else:
            return final_frags,end

    def get_graph(self,frags):
        pair = {}
        for idx,frag in enumerate(frags):
            for at in frag.GetAtoms():
                if at.GetIsotope(): # chain case (0) & spiro case (1)
                    n = at.GetIsotope()
                    if at.GetSymbol() =='*':
                        n=str(n)+'*'
                    if n not in pair:
                        # pair[n] = 
                        if at.GetSymbol() =="*":
                            pair[n] = {"type":0,"pairs":[]}
                        else:
                            pair[n] = {"type":1,"pairs":[]}
                    pair[n]['pairs'].append((idx,at.GetIdx()))
        
        for idx,frag in enumerate(frags): # plain case small 2 small
            
            for bond in frag.GetBonds():
                if bond.HasProp("idx"):
                    b,e = bond.GetBeginAtom(),bond.GetEndAtom()
                    if b.GetAtomMapNum() > e.GetAtomMapNum():
                        b,e = e,b

                    if str(bond.GetProp("idx")) not in pair:

                        pair[str(bond.GetProp("idx"))] =  {"type":2,"pairs":[]}
                    
                    # bond_map[(b.GetAtomMapNum(),e.GetAtomMapNum(),bond.GetProp("idx"))].append(idx)
                    pair[str(bond.GetProp("idx"))]['pairs'].append((idx,(b.GetIdx(),e.GetIdx())))
        
        for bond_id in pair:
            if pair[bond_id]["type"] == 2:
                pair[bond_id]["type"] = 2 if get_plain_iface_type(pair[bond_id]["pairs"][0][1], pair[bond_id]["pairs"][1][1]) else 3
        
        edge_index = [[],[]]
        atom_map = [[],[]]
        iface_type = []
        for x in pair:
            if len(pair[x]["pairs"]) == 2:
                edge_index[0].append(pair[x]["pairs"][0][0])
                edge_index[1].append(pair[x]["pairs"][1][0])
                if isinstance(pair[x]["pairs"][0][1],int):
                    atom_map[0].append(pair[x]["pairs"][0][1])
                    atom_map[1].append(pair[x]["pairs"][1][1])
                else:
                    atom_map[0].append(sort_atoms(pair[x]["pairs"][0][1]))
                    atom_map[1].append(sort_atoms(pair[x]["pairs"][1][1]))
                iface_type.append(pair[x]["type"])
                # final_pair[x] = pair[x]
        return {"edge_index":edge_index,"iface_atoms":atom_map,"iface_type":iface_type}
