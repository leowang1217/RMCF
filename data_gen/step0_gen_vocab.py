from processor.frag_processor import FragProcessor
import pickle
import multiprocessing as mp
from tqdm import tqdm

smi_map = pickle.load(open('data_gen/smi_map.pkl','rb'))

mols = []
def process_mol(smi):
    mols = []
    paths = smi_map[smi][1]
    for p in paths:
        d = pickle.load(open(f'rdkit_folder/{p}','rb'))
        mols+=[conf['rd_mol'] for conf in d['conformers']]
    return mols

smis = [smi for smi in smi_map if not smi_map[smi][0]]
total = []
with mp.Pool(40) as p:
    results = tqdm(p.imap_unordered(process_mol,smis), total=len(smis))
    for mols in results:
        total+=mols
processor = FragProcessor()
processor.bulid(total)