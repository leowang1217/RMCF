
from rdkit.Chem.rdMolAlign import AlignMol
from rdkit import Chem
import numpy as np
import math 
from utils.fragment import reorder_frag,get_clean_smi,wash_frag
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from copy import deepcopy
from segmentor.brics_seg import Single_Bond_Seg
from segmentor.cycle_seg import CycleSeg
from segmentor.chain_seg import ChainSeg
from segmentor.brics_seg import BRICS_RING_R_Seg
import random
import multiprocessing as mp
from functools import partial
class FragVocab:
    def __init__(self, vocab_file=None):
        self.frag_ph = 10
        self.conf_ph = 30
        if vocab_file:
            # vocab = pickle.load(open(vocab_file,'rb'))
            self.load_vocab(vocab=vocab_file)

    

    def load_vocab(self,vocab):
        print("Load Vocab")
        vocab_2d = vocab.keys()
        
        self.vocab_3d = []
        count = 0
        conf_map = []
        for frags in vocab.values():
            self.vocab_3d+= frags
            conf_map.append(list(range(count,count+len(frags))))
            count+=len(frags)

        self.smi2idx = {}
        self.idx2smi = {}
        self.frag2conf = {}
        self.conf2frag ={}
        self.iface_map = {}
        self.frag2iface = {}
        self.iface2frag = {}
        self.iface_id = 20
        for idx,smi in enumerate(tqdm(vocab_2d)):
            frag_idx = idx + self.frag_ph
            self.smi2idx[smi] = frag_idx # 10 placeholder 
            self.idx2smi[frag_idx] = smi
            self.iface_map[frag_idx] = self.get_ifaces(self.vocab_3d[conf_map[idx][0]])
            self.frag2iface[frag_idx] =[]
            for iface_idx in self.iface_map[frag_idx]['idx2iface'].keys():
                self.frag2iface[frag_idx].append(iface_idx)
                self.iface2frag[iface_idx]= frag_idx
                # max_iface_index = max(max_iface_index,iface_idx)
            self.frag2conf[frag_idx] = []
            for conf_idx in conf_map[idx]:
                conf_idx+=self.conf_ph
                self.frag2conf[frag_idx].append(conf_idx)
                self.conf2frag[conf_idx] = frag_idx 
            # self.frag_index_to_iface_atoms[vocab_2d[smi][0]] = get_ifaces((frag))
    def get_2d_vocab_size(self):
        return len(self.smi2idx) + self.frag_ph
    def get_iface_size(self):
        return self.iface_id -1
    def get_3d_vocab_size(self):
        return len(self.vocab_3d) + self.conf_ph

    def SmilesToFrag(self,smi):
        return self.smi2idx[smi]
    def FragToSmiles(self,frag_index):
        return self.idx2smi[frag_index]
    def FragToConfs(self,frag_index):
        return self.frag2conf[frag_index]
    def SmilesToConfs(self,smi):
        return self.frag2conf[self.smi2idx[smi]]
    def ConfToFrag(self,conf_index):
        return self.conf2frag[conf_index]
    def FragAtomToIface(self,frag_index,atoms):
        # print(self.iface_map[frag_index]["atom2idx"])
        # print(self.FragToSmiles(frag_index))
        try:
            iface = self.iface_map[frag_index]["atom2idx"][atoms]
        except:
            iface = list(self.iface_map[frag_index]["atom2idx"].values())[0]
            print('ERROR_KEY',frag_index,atoms)
        return iface
    def IfaceToDihAtoms(self,frag_index,iface_index):
        try:
            atoms = self.iface_map[frag_index]["idx2iface"][iface_index]
        except:
            print(frag_index,iface_index)
            
        return atoms
    def FragToMols(self,frag_index):
        return [(i,self.vocab_3d[i-self.conf_ph]) for i in self.frag2conf[frag_index]]
    def ConfToMol(self,conf_index):
        return self.vocab_3d[conf_index-self.conf_ph]
    # def conf_to_dihedral_atoms(self,conf_index,iface_index):
    #     frag_index = self.conf_to_frag(conf_index)
    #     return self.iface_index_to_dihedral_atoms(frag_index,iface_index)
    def ConfToDihAtoms(self,conf_index,iface_index):
        frag_index = self.ConfToFrag(conf_index)
        return self.IfaceToDihAtoms(frag_index, iface_index)
    def FragToSmiles(self,frag_index):
        return self.idx2smi[frag_index]
    def FragToIfaces(self,frag_index):
        return self.frag2iface[frag_index]
    def IfaceToFrag(self,iface_index):
        return self.iface2frag[iface_index]

        
    def get_ifaces(self,frag):
        
        atom2idx = {}
        idx2iface = {}
        ri = frag.GetRingInfo()
        cycles = [list(x) for x in ri.AtomRings()]
        # chain_iface_count = 0
        # spiro_iface_count = 0
        # plain_iface_count = 0
        # iface_idx = 0

        for at_idx,at in enumerate(frag.GetAtoms()):
            if at.GetSymbol() =='*':
                # iface2idx[at.GetIdx()] = chain_iface_count
                atom2idx[at.GetIdx()] = self.iface_id
                if len(at.GetNeighbors())>1:
                    print('ERROR3',Chem.MolToSmiles(frag))
                bridge = at.GetNeighbors()[0]
                idx2iface[self.iface_id] = [at.GetIdx(),bridge.GetIdx()]
                # iface2atoms[at.GetIdx()] = [at.GetIdx(),bridge.GetIdx()]
                for at2 in bridge.GetNeighbors():
                    if at2.GetIdx()!= at_idx:
                        idx2iface[self.iface_id].append(at2.GetIdx())
                        # iface2atoms[at.GetIdx()].append(at2.GetIdx())
                        break
                # chain_iface_count+=1
                self.iface_id+=1
            elif at.IsInRing():
                for nbr in at.GetNeighbors():
                    if nbr.IsInRing() and nbr.GetIdx() > at.GetIdx():
                        at_idx,nbr_idx = at.GetIdx(),nbr.GetIdx()
                        ring_nbr_atoms = []
                        for cycle in cycles:
                            if at_idx in cycle and nbr_idx in cycle:
                                ring_nbr_atoms = [at for at in cycle if at!=at_idx and at!= nbr_idx ]
                        idx2iface[self.iface_id] = [at_idx,nbr_idx]
                        atom2idx[(at.GetIdx(),nbr.GetIdx())] = self.iface_id
                        for at_nbr in at.GetNeighbors():
                            if at_nbr.GetIdx() in ring_nbr_atoms:
                                idx2iface[self.iface_id].append(at_nbr.GetIdx())
                                break
                        for nbr_nbr in nbr.GetNeighbors():
                            if nbr_nbr.GetIdx() in ring_nbr_atoms:
                                # iface2atoms[(at.GetIdx(),nbr.GetIdx())].append(nbr_nbr.GetIdx())
                                idx2iface[self.iface_id].append(nbr_nbr.GetIdx())
                                break
                        # plain_iface_count +=1
                        self.iface_id+=1

                # idx2iface[self.iface_id] = [at.GetIdx()]+[nbr.GetIdx() for nbr in  at.GetNeighbors() if nbr.IsInRing()][:2]
                # atom2idx[at.GetIdx()] = self.iface_id
                # self.iface_id+=1
                if (at.GetIsAromatic()) or (at.GetSymbol() in ('O')) or (at.GetSymbol() =='C' and at.GetDegree()>2):
                    continue
                nbrs = [nbr.GetIdx() for nbr in  at.GetNeighbors() if nbr.IsInRing()][:2]
                nbrs.sort()
                idx2iface[self.iface_id] = [at.GetIdx()]+nbrs
                atom2idx[at.GetIdx()] = self.iface_id 
                self.iface_id+=1
        return {"atom2idx":atom2idx,"idx2iface":idx2iface}
                    
    def build_vocab(self,training_mols):
        brics_ring_sg = BRICS_RING_R_Seg()
        cycle_sg = CycleSeg()
        single_bond_sg = Single_Bond_Seg()
        chain_sg = ChainSeg()
        frags = {}
        random.shuffle(training_mols)
        print('Virtual Fragmentation')

        with mp.Pool(40) as p:
            results = tqdm(p.imap_unordered(partial(get_frags,
                                        brics_ring_sg= brics_ring_sg,
                                        cycle_sg = cycle_sg,
                                        single_bond_sg = single_bond_sg
                                        ),training_mols),total=len(training_mols))
            for fs,smis in results:
                for frag,frag_smi in zip(fs,smis):
                    if frag_smi not in frags:
                        frags[frag_smi] = []
                    if len(frags[frag_smi])<1000:
                        frags[frag_smi].append(frag)


        print('Clustering')
        chain_score = {}
        vocab = {}
        chain_vocab = {}
        mols = list(frags.values())
        # smis = list(frags.keys())
        with mp.Pool(40) as p:
            results = tqdm(p.imap_unordered(best_cluster,mols), total=len(mols))
            for var,centers in results:
                # print(var,centers)
                smi = get_clean_smi(centers[0])
                if not len(Chem.GetSSSR(centers[0])):
                    chain_score[smi] = var
                    chain_vocab[smi] = centers
                else:
                    vocab[smi] = centers
        #     for x in results:
        #         x
        print("Shrinking")
        sg_hit_vocab = chain_sg.gen_vocab(chain_score)
        atomic = set()
        chain_sg = ChainSeg(hit_vocab=sg_hit_vocab)
        for smi in tqdm(chain_score):
            mol = Chem.MolFromSmiles(smi)
            frags,_ = chain_sg.fragmentize(mol)
            for f in frags:
                atomic.add(get_clean_smi(f))
        for smi in atomic:
            if smi not in chain_vocab:
                for p,p_smi,_ in chain_sg.rigid_frags:
                    mol = Chem.MolFromSmiles(smi)
                    flag = chain_sg.has_substruct(mol,p)
                    if flag:
                        frags,_ = chain_sg.remove_core(mol,p)
                        valid  = chain_sg.is_valid(frags)
                        if valid and not frags:
                            vocab[smi] = chain_vocab[p_smi]
                            break
            else:
                vocab[smi] = chain_vocab[smi]
        return vocab,sg_hit_vocab
    





def gen_dist_map(mols):
    if len(mols)<1:
        return []
    mols = [reorder_frag(wash_frag(x)) for x in mols]
    num_atoms = range(len(mols[0].GetAtoms()))
    num_mols = len(mols)
    dist_map=[]
    for i in range(num_mols-1):
        # rmsd_mol.append([])
        dist_map.append([])
        for j in range(i+1,num_mols):
            dist_map[i].append(AlignMol(mols[i],mols[j],atomMap=[(i,i) for i in num_atoms]))
    return dist_map



def clustering(d,n):
    # n = get_center_num(d)
    nums = len(d)+1
    # n = get_center_num(nums)
    pred_dst =[]
    for dd in d:
        pred_dst+=dd
    pred_dst = np.array(pred_dst)
    pred_adj = np.zeros((nums, nums))
    pred_adj[np.triu(np.ones((nums, nums)),1) == 1] = pred_dst
    pred_adj = pred_adj.T + pred_adj
    if n==1:
        d = np.sum(np.square(pred_adj),axis=1)
        idx = np.argmin(d)
        return [idx],d[idx]/nums,0
    model = KMedoids(n_clusters=n, metric='precomputed', method='pam', init='heuristic', max_iter=300, random_state=None)
    model.fit(pred_adj)
    try:
        score=silhouette_score(pred_adj, model.labels_,metric='precomputed', sample_size=None, random_state=None,)
    except ValueError:
        score=0.0
    
    variance = np.sum(np.square(np.min(pred_adj[:,model.medoid_indices_], axis=1)))/nums
    return model.medoid_indices_.tolist(),variance,score


def best_cluster(frags):
    num_atoms = len(frags[0].GetAtoms())
    dist_map = gen_dist_map(frags)
    if not dist_map:
        return 100,[reorder_frag(wash_frag(frags[0]))]
    max_cluster_num = min(max(math.floor(math.sqrt((len(frags)-1)/10)),1)+1,12)
    # smi = get_clean_smi(frags[0])
    centers,variances,scores = [],[],[]
    for n in range(1,max_cluster_num+1):
        center,variance,score = clustering(dist_map,n)
        centers.append(center)
        variances.append(variance)
        scores.append(score)
    idx = np.argmax(np.array(scores))
    if variances[idx] > 0.001*num_atoms:
        return variances[idx]*num_atoms,[reorder_frag(wash_frag(frags[i])) for i in centers[idx]]
        
    else:
        for c,v in zip(centers,variances):
            if v<= 0.001*num_atoms:
                return variances[idx]*num_atoms,[reorder_frag(wash_frag(frags[i])) for i in c]

def get_frags(mol,brics_ring_sg,cycle_sg,single_bond_sg):
        # for mol in tqdm(training_mols):
    try:
        smi = Chem.MolToSmiles(mol)

        if "." in smi:
            return [],[]
            # continue
        Chem.SanitizeMol(mol) 
                        
        mol = Chem.rdmolops.RemoveAllHs(mol, sanitize=True)
        assert mol.GetNumHeavyAtoms() == mol.GetNumAtoms(), "cannot remove all Hs for %s" % smi
        brics_frags,_ = brics_ring_sg.fragmentize(mol)
        # cycle_frags = []
        # rigid_frags = []
        frags = []
        for frag in brics_frags:
            if len(Chem.GetSSSR(frag)):
                # cycles_frag = deepcopy(frag)
                cycle,_ = cycle_sg.fragmentize(deepcopy(frag))
                frags+=cycle
                # cycle_frags+=cycle
            else:
                # chain_frag = deepcopy(frag)
                frags.append(frag)
                rigid,_ = single_bond_sg.fragmentize(deepcopy(frag))
                frags+=rigid
        # frags = cycle_frags + rigid_frags + brics_frags
        smis = [get_clean_smi(x) for x in frags]
        return frags,smis
    except Exception as e:
        print(e)
        return [],[]
