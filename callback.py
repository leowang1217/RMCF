import json
import logging
import numpy as np
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from torch.utils.data import DataLoader
from torch_geometric.loader.dataloader import Collater
from tqdm import tqdm
import pickle
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)

class RMCFCallback(TrainerCallback):
    
    def __init__(self, 
                trainer,
                config,
                eval_rank,
                ):
        self.trainer = trainer
        self.eval_steps = config.eval_steps
        self.model = trainer.model
        self.eval_dataloader = DataLoader(
            trainer.eval_dataset, 
            batch_size=self.trainer.args.per_device_eval_batch_size,
            collate_fn=Collater(None,None)
            )
        self.eval_rank = eval_rank
        # self.thres = config.cov_thres
        # self.sampling_strategy = config.sampling_strategy
        mols = pickle.load(open(f'{config.valid_prefix}_mols.pkl','rb'))
        valid_data = [json.loads(l) for l in open(f'{config.valid_prefix}.json','r').readlines()]
        self.ref_mols = {}
        for valid in valid_data:
            smi = valid['smiles']
            self.ref_mols[smi] = {
                'mols': mols[smi],
                'edge_index' : valid['edge_index'],
                'iface_index' : valid['iface_index'],
                'iface_types' :valid['iface_types']
            }

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step%self.eval_steps==0 and state.global_step:
            self.model.eval()
            if self.trainer.args.local_rank == self.eval_rank:
                covmat = {"MAT-P":[],"COV-P":[],"MAT-R":[],"COV-R":[]}
                for batch in tqdm(self.eval_dataloader):
                    batch = self.trainer._prepare_inputs(batch)
                    results = self.model.evaluate(data=batch,
                                                ref_mols = self.ref_mols
                                                )
                    for m in results.keys():
                        covmat[m].extend(results[m])
                print(f'\nEval Steps:{state.global_step}')
                for metric in ['COV-R','MAT-R','COV-P','MAT-P']:
                    print(f"{metric}: MEAN {np.mean(np.array(covmat[metric]))} MEDIAN {np.median(np.array(covmat[metric]))}")
                print()
            self.model.train()

