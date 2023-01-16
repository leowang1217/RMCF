
from torch.nn.modules.loss import CrossEntropyLoss
from models import register_model
import torch.nn as nn
import torch
import numpy as np
import torch.multiprocessing as mp

from models.modules.mpnn import GAT
from torch_geometric.utils import unbatch_edge_index
from torch_random_fields.models.general_crf import GeneralCRF
from torch_random_fields.utils.gibbs import gibbs_sampling
from utils.covmat import covmat
from tqdm import tqdm
from functools import partial
from sklearn.cluster import KMeans
@register_model("rmcf")
class RMCF(nn.Module):
    
    def __init__(self, config,processor=None):
        super().__init__()
        self.config = config
        self.hidden_dim = config.dim_h
        self.node_dim = config.dim_node
        self.edge_dim = config.dim_edge
        self.mpnn_steps = config.mpnn_steps
        self.vocab_2d_num = config.vocab_2d_size
        self.vocab_3d_num = config.vocab_3d_size
        self.n_max_states = int(360/config.angle_intervals)
        self.node_embedding = nn.Embedding(self.vocab_2d_num + self.n_max_states,config.dim_node)
        self.node_type_embedding = nn.Embedding(2,config.dim_node)
        self.iface_embedding = nn.Embedding(config.iface_size + self.n_max_states ,config.dim_edge)
        self.mol_encoder = self.build_mol_encoder(config)
        # self.n_max_states = int(360/config.angle_intervals)
        self.angle_mask = list(range(self.n_max_states))
        self.node_output  = nn.Sequential(nn.Linear(config.dim_h,self.vocab_3d_num+self.n_max_states*2))          
        self.node_loss_fct = CrossEntropyLoss()
        self.random_fields = GeneralCRF(num_states=self.n_max_states,feature_size= config.dim_h)
        self.sampling_strategy = config.sampling_strategy
        self.cov_thres = config.cov_thres
        self.processor = processor

        
    def build_mol_encoder(self, config):
        return GAT(node_input_dim=config.dim_node,
                    edge_input_dim=config.dim_edge,
                    output_dim= config.dim_h,
                    attn_heads=config.num_attn_heads
                    )

    def encode_mol(self, data):
        x  = self.node_embedding(data.x.to(torch.int)) + self.node_type_embedding(data.node_type.to(torch.int))
        edge_attr =  self.iface_embedding(data.edge_attr.to(torch.int))
        graph_hidden = self.mol_encoder.forward(x = x, edge_attr = edge_attr, edge_index = data.edge_index)
        return graph_hidden

    def forward(self, data):
        # print(data.mask,data.x_tag)
        
        graph_hidden = self.encode_mol(data)
        # node_hidden = graph_hidden[data.node_mask]
        hidden_num = len(graph_hidden)
        node_logit = self.node_output(graph_hidden)
        pred_mask = self.mask_node(data,hidden_num)
        start  = get_start_pos(masks=data.mask,hnum_edges=data.hnum_edges)
        unaries = unaries_shift(unaries=node_logit+pred_mask,start=start,shift_size=self.n_max_states)
        targets = target_shift(data.x_tag,start)
        loss = self.random_fields.forward(unaries = unaries.unsqueeze(0),
                                          binary_edges=data.edge_index.unsqueeze(0),
                                          targets = targets.unsqueeze(0),
                                          node_features=graph_hidden.unsqueeze(0)
                                          )
        return loss,{"node_loss":loss}
                
    @torch.no_grad()
    
    def evaluate(self,
                 data,
                 ref_mols,
                 return_mols = False
                 ):
        metrics = {"MAT-P":[],"COV-P":[],"MAT-R":[],"COV-R":[]}
        mols = []
        node_split = (data.hnum_edges+data.hnum_nodes).tolist()
        edge_split = (data.hnum_edges*4).tolist()

        graph_hidden = self.encode_mol(data)
        hidden_num = len(graph_hidden)
        node_logit = self.node_output(graph_hidden)
        pred_mask = self.mask_node(data,hidden_num)
        st  = get_start_pos(masks=data.mask,hnum_edges=data.hnum_edges)
        
        unaries = unaries_shift(unaries=node_logit + pred_mask,start=st, shift_size=self.n_max_states)
        pred_mask = unaries_shift(unaries=pred_mask,start=st, shift_size=self.n_max_states)
        
        potentials = self.random_fields.forward(unaries = unaries.unsqueeze(0),
                                                binary_edges=data.edge_index.unsqueeze(0),
                                                node_features=graph_hidden.unsqueeze(0),
                                                return_potentials=True)
        unary_potentials = torch.split(potentials['unary'].squeeze().detach().cpu(),node_split)
        binary_potentials = torch.split(potentials['binary'].squeeze().detach().cpu(),edge_split)
        binary_edges = unbatch_edge_index(data.edge_index.detach().cpu(),data.batch.detach().cpu())
        targets = torch.split(potentials['targets'].squeeze().detach().cpu(),node_split)
        logit_masks = torch.split(pred_mask.detach().cpu().gather(-1, potentials['targets'].squeeze().detach().cpu()),node_split)
        start = torch.split(torch.tensor(st,requires_grad=False,dtype=torch.long),node_split)
        
        paras = [{'unary':unary_potentials[i],
                'binary':binary_potentials[i],
                'edges': binary_edges[i],
                'masks': logit_masks[i],
                'targets':targets[i],
                'start': start[i],
                'sample_size':int(data.num_conf[i]),
                'num_nodes':int(data.hnum_nodes[i]),
                'ref':ref_mols[data.smiles[i]],
                'smiles':data.smiles[i]
                } for i in range(len(unary_potentials))]
        
        with mp.Pool(40) as p:
            results = tqdm(p.imap_unordered(partial(asm_mol,
                                        sampling_strategy=self.sampling_strategy,
                                        processor=self.processor,
                                        cov_thres = self.cov_thres,
                                        return_mols = return_mols,
                                        n_max_states = self.n_max_states),paras),total=len(paras))
            if return_mols:
                for confs in results:
                    mols.append(confs)
            else:
                for metric in results:
                    # print(metric)
                    if metric['COV-P']>=0:
                        for m in metrics.keys():
                            metrics[m].append(metric[m])
        return mols if return_mols else metrics
    
    def mask_node(self,data,node_num):
        # mask = list(chain.from_iterable(data.mask))
        node_mask = torch.full((node_num,self.vocab_3d_num+self.n_max_states*2),fill_value=-1000000,dtype=int,device=data.x.device)
        idx  = 0
        idx_pairs = []
        cur_batch = 0
        # cur_edge = 0
        for num_nodes,num_edges in zip(data.hnum_nodes.int().tolist(),data.hnum_edges.int().tolist()):
            for i in range(num_nodes):
                idx_pairs +=[(idx,j) for j in data.mask[cur_batch][i]]
                idx+= 1
            for _ in range(num_edges):
                idx_pairs += list(zip([idx]*self.n_max_states,self.angle_mask))
                idx+=1
            cur_batch +=1
        
        ind = torch.tensor(idx_pairs,device=data.x.device,requires_grad=False)
        logit = node_mask.view(-1)
        return logit.scatter(0, ind[:, 0] * node_mask.shape[1] + ind[:, 1], torch.zeros_like(logit,device=data.x.device)).reshape(node_mask.shape)

def get_start_pos(masks,hnum_edges):
    start = []
    for batch_idx,num_edges in enumerate(hnum_edges.int().tolist()):
        start +=[x[0] for x in masks[batch_idx]]
        start +=[0]*num_edges
    return start

def target_shift(targets,start):
    new_targets = targets - torch.tensor(start,requires_grad=False).int().to(targets.device)
    return new_targets

def index_recover(pred_idx,start):
    return pred_idx + torch.tensor(start).int().to(pred_idx.device)

def unaries_shift(unaries,start,shift_size):
        idx = []
        for x in start:
            idx.append(list(range(x,x+shift_size)))
        return torch.gather(unaries,1,torch.tensor(idx,requires_grad=False).long().to(unaries.device))

def clustering(feature,output_size):
    X = np.array(feature)
    kmeans = KMeans(n_clusters=output_size, random_state=0).fit(X)
    labels = kmeans.labels_
    label_set= set()
    output_idx = []
    for idx,l in enumerate(labels):
        if l not in label_set:
            output_idx.append(idx)
            label_set.add(l)
    while len(output_idx) < output_size:
        output_idx += output_idx[:output_size-len(output_idx)]
    return output_idx

def asm_mol(para,
                processor,
                n_max_states,
                sampling_strategy,
                cov_thres,
                return_mols = False):
    sample_size = para['sample_size'] * 2
    samples = gibbs_sampling(unary_potentials=para['unary'],
                binary_potentials= para['binary'],
                binary_edges=para['edges'],
                logit_mask=para['masks'],
                sample_size= 10000 if sampling_strategy =='clustering' else sample_size
    )
    pred_idx = torch.tensor(samples, requires_grad=False).long()
    pred_idx = para['targets'].unsqueeze(0).expand(10000 if  sampling_strategy =='clustering' else sample_size,-1,-1).gather(-1,pred_idx.unsqueeze(-1)).squeeze(-1) +para['start']
    frag_list = [(x[:para['num_nodes']]- n_max_states).tolist() for x in pred_idx]
    dih_list = [x[para['num_nodes']:].tolist() for x in pred_idx]
    idxs = clustering(dih_list,output_size=sample_size) if  sampling_strategy =='clustering' else range(len(frag_list))
    ref = para['ref']
    pred_mols = []
    for j in idxs:
        asm_mol = processor.assemble_mol(conf_index = frag_list[j], 
                                            dihedrals = dih_list[j], 
                                            edge_index = ref['edge_index'],
                                            iface_index = ref['iface_index'], 
                                            iface_types = ref['iface_types'])
        pred_mols.append(asm_mol)
    if return_mols:
        return pred_mols
    else:
        print('Calc COVMAT',para['smiles'],flush=True)
        metric = covmat(pred_mols=pred_mols,ref_mols=ref['mols'],thres=cov_thres)
        # print('Calc COVMAT',para['smiles'],time.time()-time1)
        return metric