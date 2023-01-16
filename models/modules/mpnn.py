import torch
import torch.nn as nn
from torch_geometric.nn import GATConv,HEATConv

class GAT(torch.nn.Module):
    def __init__(self,
                 node_input_dim=15,
                 edge_input_dim=5,
                 output_dim = 300,
                 node_hidden_dim = 300,
                 attn_heads = 8,
                 num_step_message_passing=6):
                 
        super(GAT, self).__init__()
        self.num_step_message_passing = num_step_message_passing
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)
        self.dropouts1 = []
        for _ in range(num_step_message_passing):
            self.dropouts1.append(nn.Dropout(0.1))
        self.dropouts1 = nn.ModuleList(self.dropouts1)

        self.dropouts2 = []
        for _ in range(num_step_message_passing):
            self.dropouts2.append(nn.Dropout(0.1))
        self.dropouts2 = nn.ModuleList(self.dropouts2)
        
        self.gat = GATConv(in_channels = node_input_dim,edge_dim=edge_input_dim,out_channels=int(output_dim//attn_heads),heads=attn_heads)
        self.ffn = []
        self.ln1 = []
        for _ in range(num_step_message_passing):
            self.ffn.append(nn.Sequential(
            nn.Linear(output_dim, output_dim * 4), nn.ReLU(),
            nn.Linear(output_dim * 4,output_dim)))
        self.ffn = nn.ModuleList(self.ffn)
        self.ln1 = []
        for _ in range(num_step_message_passing):
            self.ln1.append(nn.LayerNorm(output_dim,device=None))
        self.ln1 = nn.ModuleList(self.ln1)
        self.ln2 = []
        for _ in range(num_step_message_passing):
            self.ln2.append(nn.LayerNorm(output_dim,device=None))
        self.ln2 = nn.ModuleList(self.ln2)

    def forward(self,x, edge_index,edge_attr=None):
        m = x
        for i in range(self.num_step_message_passing):
            m = self.ln1[i](self.dropouts1[i](self.gat(x= m,edge_index=edge_index,edge_attr=edge_attr))+m)
            m = self.ln2[i](self.dropouts2[i](self.ffn[i](m))+m)
            
        return m


class HGAT(torch.nn.Module):
    def __init__(self,
                 node_input_dim=15,
                 output_dim = 300,
                 attn_heads = 8,
                 num_step_message_passing=6):
                 
        super(HGAT, self).__init__()
        self.num_step_message_passing = num_step_message_passing
        self.hgat = HEATConv(in_channels = node_input_dim, out_channels=int(output_dim//attn_heads),num_node_types=2, num_edge_types=3, edge_type_emb_dim=10, edge_dim =1 , edge_attr_emb_dim= 50, heads= attn_heads, root_weight = False)
        self.ffn = nn.ModuleList([nn.Sequential(
            nn.Linear(output_dim, output_dim * 4), nn.ReLU(),
            nn.Linear(output_dim * 4,output_dim)) for _ in range(num_step_message_passing) ])
        
        self.ln1 = nn.ModuleList([nn.LayerNorm(output_dim,device=None) for _ in range(num_step_message_passing)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(output_dim,device=None) for _ in range(num_step_message_passing)])
        self.dp1 = nn.ModuleList([nn.Dropout(0.1) for _ in range(num_step_message_passing)])
        self.dp2 = nn.ModuleList([nn.Dropout(0.1) for _ in range(num_step_message_passing)])
        

    def forward(self,data,x):
        m = x
        for i in range(self.num_step_message_passing):
            m = self.ln1[i](self.dp1[i](self.hgat(x= m,edge_index=data.edge_index,edge_attr=data.edge_attr.unsqueeze(1),node_type=data.node_type,edge_type=data.link_type))+m)
            m = self.ln2[i](self.dp2[i](self.ffn[i](m))+m)
        return m