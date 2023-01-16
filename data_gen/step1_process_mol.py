import pickle
from tqdm import tqdm
import json
import math
import multiprocessing as mp
from rdkit import Chem

smi_map = pickle.load(open('data_gen/smi_map.pkl','rb'))


from segmentor.frag_seg import FragSeg
from data.vocabulary import FragVocab
from processor.frag_processor import FragProcessor
hit_vocab = pickle.load(open('geom-drugs/hit.pkl','rb'))
vocab_file = pickle.load(open('geom-drugs/vocab.pkl','rb'))
seg = FragSeg(hit_vocab=hit_vocab)
vocab = FragVocab(vocab_file=vocab_file)
processor  = FragProcessor(segmentor= seg, vocab =vocab )

count = 0
def process_mol(smi):
    paths = smi_map[smi][1]
    split = smi_map[smi][0]
    mols = {}
    if split:
        output = {}
    else:
        output = []

    for p in paths:
        d = pickle.load(open(f'rdkit_folder/{p}','rb'))
        
        
        for conf in d['conformers']:
            mol = conf['rd_mol']
            try:
                mol = processor.sanitize(mol)
                if not mol:
                    continue
                info = processor.fragmentize_sanitized_mol(mol)
                if not info:
                    print('ERROR1')
                    continue
            except Exception as e:
                print(e)
                continue
            if not info['edge_index'][0]:
                print('cannot fragmentize')
                continue
            smi = Chem.MolToSmiles(mol,isomericSmiles=False)
            out = {
                'smiles':smi,
                'frag_index':info['frag_index'],
                "edge_index":info['edge_index'],
                "iface_index":info['iface_index'],
                "iface_types":info['iface_types'],
                # 'conf_index':info['conf_index'],
                # "dihedrals":info['dihedrals'],
                # "blzm":conf['boltzmannweight']
            }
            if split:
                if smi not in mols:
                    mols[smi] = []
                mols[smi].append(mol)
                output[smi] = out
            else:
                out['conf_index'] = info['conf_index']
                out['dihedrals'] = [0 if math.isnan(d) else d  for d in info['dihedrals']]
                out['blzm'] = conf['boltzmannweight']
                output.append(out)
    return {
            'split':split,
            'info':output,
            'mols':mols
        }
        # return output,mols
    # else:
    #     return None,None

train = open('geom-drugs/drugs.json','w')
valid = open('geom-drugs/valid.json','w')
test = open('geom-drugs/test.json','w')
valid_set = {}
test_set = {}
test_mol_map = {}
valid_mol_map = {}
# smis = [smi for smi in smi_map.keys() if smi_map[smi][0]!=0]
# for smi in tqdm(smis):
#     process_mol(smi)
with mp.Pool(40) as p:
    results = tqdm(p.imap_unordered(process_mol,list(smi_map.keys())), total=len(list(smi_map.keys())))
    # results = tqdm(p.imap_unordered(process_mol,smis), total=len(smis))
    for data in results:
        split  = data['split']
        if not split:
            for x in data['info']:
                train.write(json.dumps(x)+'\n')
            
        else:
            # print(data)
            if split == 1:
                for smi in data['mols']:
                    if smi not in test_mol_map:
                        test_mol_map[smi] = []
                    test_mol_map[smi].extend(data['mols'][smi])
                    test_set[smi] = data['info'][smi]
            elif split == 2:
                for smi in data['mols']:
                    if smi not in valid_mol_map:
                        valid_mol_map[smi] = []
                    valid_mol_map[smi].extend(data['mols'][smi])
                    valid_set[smi] = data['info'][smi]
            # print(len(mol_map))
pickle.dump(test_mol_map,open('geom-drugs/test_mols.pkl','wb'))
pickle.dump(valid_mol_map,open('geom-drugs/valid_mols.pkl','wb'))
for smi in tqdm(test_set):
    out = test_set[smi]
    out['num_conf'] = len(test_mol_map[smi])
    test.write(json.dumps(out)+'\n')
for smi in tqdm(valid_set):
    out = valid_set[smi]
    out['num_conf'] = len(valid_mol_map[smi])
    valid.write(json.dumps(out)+'\n')


    