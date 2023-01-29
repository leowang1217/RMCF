import torch 
import os
import argparse
import pickle
import json
import numpy as np
from rdkit import Chem
from models import setup_model
from transformers.utils import logging
from utils.file_access import load_state_dict_in_model, prepare_inputs
from train import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch_geometric.loader.dataloader import Collater
from segmentor.frag_seg import FragSeg
from data.vocabulary import FragVocab
from processor.frag_processor import FragProcessor
logger = logging.get_logger(__name__)



@torch.no_grad()
def test(args):
    seg = FragSeg(hit_vocab=pickle.load(open(f'{args.seg_vocab_path}','rb')))
    vocab = FragVocab(vocab_file=pickle.load(open(f'{args.vocab_path}','rb')))
    processor  = FragProcessor(segmentor= seg, vocab =vocab)
    args.vocab_2d_size = vocab.get_2d_vocab_size()
    args.vocab_3d_size = vocab.get_3d_vocab_size()
    args.iface_size = vocab.get_iface_size()

    test_dataset = load_dataset(args,vocab,split="test")
    mols = pickle.load(open(f'{args.test_prefix}_mols.pkl','rb'))
    test_data = [json.loads(l) for l in open(f'{args.test_prefix}.json','r').readlines()]
    ref_mols = {}
    for data in test_data:
        smi = data['smiles']
        ref_mols[smi] = {
            'mols': mols[smi],
            'edge_index' : data['edge_index'],
            'iface_index' : data['iface_index'],
            'iface_types' :data['iface_types']
        }

    
    last_checkpoint =args.model_dir
    state_dict = torch.load(os.path.join(last_checkpoint, "pytorch_model.bin"), map_location="cuda:0")
    model = load_state_dict_in_model(setup_model(args,processor= processor),state_dict)
    model = model.cuda()
    del state_dict
    logger.info("Model loaded")

    test_dataloader = DataLoader(test_dataset, 
                                batch_size=args.batch_size,
                                collate_fn=Collater(None,None),
                                shuffle=False)
    
    covmat = {"MAT-P":[],"COV-P":[],"MAT-R":[],"COV-R":[]}
    total_mols = {}
    model.eval()
    for batch in tqdm(test_dataloader):
        batch = prepare_inputs(batch)
        results = model.evaluate(data=batch,
                                ref_mols = ref_mols,return_mols=args.return_mols)
        if args.return_mols:
            for m in results:
                smi = Chem.MolToSmiles(m[0])
                total_mols[smi]= m
        else:
            for m in results.keys():
                covmat[m].extend(results[m])
            for metric in ['COV-R','MAT-R','COV-P','MAT-P']:
                print(f"{metric}: MEAN {np.mean(np.array(covmat[metric]))} MEDIAN {np.median(np.array(covmat[metric]))}")
            print('VALID TEST NUM',len(covmat['COV-R']))
            print()
    if args.return_mols:
        pickle.dump(total_mols,open(f'{args.model_dir}/mols.pkl'))

if __name__ == '__main__':

    # TODO: 对args分组！
    parser = argparse.ArgumentParser()

    parser.add_argument('--test-prefix', type=str, default='geom-drugs/test.json')
    parser.add_argument('--model-name', type=str, default='rmcf')
    parser.add_argument('--model-dir', type=str, default='checkpoints/rmcf')
    parser.add_argument('--dim-h', type=int, default=512, help='dimension of the hidden')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--dim-node', type=int, default=512,)
    parser.add_argument('--dim-edge', type=int, default=512)
    parser.add_argument('--mpnn-steps', type=int, default=6, help='number of mpnn steps')
    parser.add_argument('--num-attn-heads', type=int, default=8)
    parser.add_argument('--angle-intervals', type=float, default=5.0)
    parser.add_argument('--sampling-strategy', type=str, default='random',choices=['random', 'clustering'])
    parser.add_argument('--cov-thres', type=float, default=1.25)
    parser.add_argument('--vocab-path', type=str,default='geom-drugs/vocab.pkl')
    parser.add_argument('--seg-vocab-path', type=str,default='geom-drugs/hit.pkl')
    parser.add_argument('--return-mols', action='store_true')
    args = parser.parse_args()

    test(args)

