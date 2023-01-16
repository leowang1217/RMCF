import torch
from torch.utils.data import IterableDataset
from torch_geometric.data import Data
import json
from utils.file_access import RandomAccessFile
from tqdm import tqdm
import math
not_torch_set =('smiles','mask')


# class MyData(Data):
#     def __inc__(self, key: str, value, *args, **kwargs):
#         if key == 'edges':
#             return self.num_nodes
#         else:
#             return super().__inc__(key, value, *args, **kwargs)

class RMCFDataset(IterableDataset):
    def __init__(self,vocab,angle_inv=5):
        self.vocab = vocab
        # with open(mol_path,'r') as f:
        #     print("Load Mol InFo")
        #     self.mol = [json.loads(l) for l in tqdm(f.readlines())]
        self.angle_inv = angle_inv
    def parse_line(self,line):
        
        angle_size = int(360/self.angle_inv)
        info = json.loads(line)
        # if 'mol' in info:
        #     mol = self.mol[info['mol']]
        # else:
        #     mol = info
        num_node  = len(info['frag_index'])
        num_edge = len(info['edge_index'][0])
        edge_e = list(range(num_node,num_node+num_edge))
        iface1,iface2=list(zip(*info['iface_index']))
        mask  = [[y + angle_size for y in self.vocab.FragToConfs(x)] for x in info['frag_index']]
        
        
        output = {
            # 'smiles':mol['smiles'],
            'x':info['frag_index']+[2]*num_edge,
            'edge_index':[info['edge_index'][0] + edge_e + edge_e + info['edge_index'][1],edge_e + info['edge_index'][1]+ info['edge_index'][0] + edge_e],
            'edge_attr':list(iface1+iface2)*2,
            'iface_types':info['iface_types']*4,
            'mask':mask,
            'hnum_nodes':num_node,
            'hnum_edges':num_edge,
            'node_type':[0] * num_node+[1] * num_edge,
            
        }

        if 'conf_index' in info:
            
            x_tag = [x + angle_size for x in info['conf_index']]+ [ int(((x+ 180+(self.angle_inv/2))%360)/self.angle_inv) if not math.isnan(x) else int(180/self.angle_inv) for x in info['dihedrals']]
            output['x_tag'] = x_tag
        else:
            output['num_conf'] = info['num_conf']
            output['smiles'] = info['smiles']
        # if len(x_tag)!= len(output['x']):
        #     print(mol)
        #     print(output)
        #     print(info)
        return output

class MultiRMCFDataset(RMCFDataset):
    def __init__(self,num_workers,load_path,vocab,angle_inv = 5):
        self.load_path=load_path
        self.num_workers=num_workers
        with open(f"{load_path}",'r') as f:
            self.len = len(f.readlines())

        self.f = RandomAccessFile(f"{load_path}")
        super(MultiRMCFDataset, self).__init__(vocab=vocab,angle_inv=angle_inv)
        
    def parse_file(self,file_path,worker_id):
        # with open(file_path,'r') as f:
            f = RandomAccessFile(f"{file_path}")
            # for i in range(worker_id):
            idx= worker_id
            l = f[idx]
            # l = f.readline()
            while l :
                line = l.strip()
                mol = Data.from_dict({k:(torch.tensor(v) if k not in not_torch_set else v) for k,v in self.parse_line(line).items()})
                yield mol
                idx += self.num_workers 
                if idx >= self.len:
                    idx = worker_id
                l = f[idx]

    def __iter__(self):
        worker_id = torch.utils.data.get_worker_info().id
        return self.parse_file(self.load_path,worker_id)


class SingleRMCFDataset(RMCFDataset):
    def __init__(self,load_path,vocab,angle_inv = 5):
        self.load_path = load_path
        super(SingleRMCFDataset, self).__init__(vocab=vocab,angle_inv=angle_inv)
    def parse_file(self,file_path):
        with open(file_path,'r') as f:
            for line in f:
                line=line.strip()
                info = self.parse_line(line)
                mol=Data.from_dict({k:(torch.tensor(v) if k not in not_torch_set else v) for k,v in info.items()})
                yield mol

    def __iter__(self):
        return self.parse_file(self.load_path)

