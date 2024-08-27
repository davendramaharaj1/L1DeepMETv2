import torch
import torch_geometric

from torch import nn
import torch.nn.functional as F

from torch_geometric.nn.conv import GraphConv, EdgeConv, GCNConv

from torch_cluster import radius_graph, knn_graph

class GraphMETNetwork(nn.Module):
    def __init__ (self, continuous_dim, cat_dim, norm, output_dim=1, hidden_dim=32, conv_depth=1):
    #def __init__ (self, continuous_dim, cat_dim, output_dim=1, hidden_dim=32, conv_depth=1):
        super(GraphMETNetwork, self).__init__()
       
        self.datanorm = norm

        self.embed_charge = nn.Embedding(3, hidden_dim//4)
        self.embed_pdgid = nn.Embedding(7, hidden_dim//4)
        #self.embed_pv = nn.Embedding(8, hidden_dim//4)
        
        self.embed_continuous = nn.Sequential(nn.Linear(continuous_dim,hidden_dim//2),
                                              nn.ELU(),
                                              #nn.BatchNorm1d(hidden_dim//2) # uncomment if it starts overtraining
                                             )

        self.embed_categorical = nn.Sequential(nn.Linear(2*hidden_dim//4,hidden_dim//2),
                                               nn.ELU(),                                               
                                               #nn.BatchNorm1d(hidden_dim//2)
                                              )

        self.encode_all = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                        nn.ELU()
                                       )
        self.bn_all = nn.BatchNorm1d(hidden_dim)
 
        self.conv_continuous = nn.ModuleList()        
        for i in range(conv_depth):
            mesg = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim))
            self.conv_continuous.append(nn.ModuleList())
            self.conv_continuous[-1].append(EdgeConv(nn=mesg).jittable())
            self.conv_continuous[-1].append(nn.BatchNorm1d(hidden_dim))

        self.output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2),
                                    nn.ELU(),
                                    nn.Linear(hidden_dim//2, output_dim)
                                   )
        self.pdgs = [1, 2, 11, 13, 22, 130, 211]

        # Intermediate variables to be removed later
        self._emb_cont = None
        self._emb_chrg = None
        self._emb_pdg = None
        self._emb_cat = None
        self._emb = None
        self._emb1 = None
        self._emb2 = None
        self._out = None

    def forward(self, x_cont, x_cat, edge_index, batch):
        # Normalize the input values within [0,1] range: pt, px, py, eta, phi, puppiWeight, pdgId, charge
        #norm = torch.tensor([1./2950., 1./2950, 1./2950, 1., 1., 1.]).to(device) 

        x_cont *= self.datanorm

        emb_cont = self.embed_continuous(x_cont)
        self._emb_cont = emb_cont.detach().clone()   

        emb_chrg = self.embed_charge(x_cat[:, 1] + 1)
        self._emb_chrg = emb_chrg.detach().clone()
        #emb_pv = self.embed_pv(x_cat[:, 2])

        pdg_remap = torch.abs(x_cat[:, 0])
        for i, pdgval in enumerate(self.pdgs):
            pdg_remap = torch.where(pdg_remap == pdgval, torch.full_like(pdg_remap, i), pdg_remap)
        emb_pdg = self.embed_pdgid(pdg_remap)
        self._emb_pdg = emb_pdg.detach().clone()

        emb_cat = self.embed_categorical(torch.cat([emb_chrg, emb_pdg], dim=1))
        #emb_cat = self.embed_categorical(torch.cat([emb_chrg, emb_pdg, emb_pv], dim=1))
        self._emb_cat = emb_cat.detach().clone()

        emb = self.bn_all(self.encode_all(torch.cat([emb_cat, emb_cont], dim=1)))
        self._emb = emb.detach().clone()
                
        # graph convolution for continuous variables
        for i, co_conv in enumerate(self.conv_continuous):
            #dynamic, evolving knn
            #emb = emb + co_conv[1](co_conv[0](emb, knn_graph(emb, k=20, batch=batch, loop=True)))
            #static
            emb = emb + co_conv[1](co_conv[0](emb, edge_index))

            if i == 0:
                self._emb1 = emb.detach().clone()
            else:
                self._emb2 = emb.detach().clone()
                
        out = self.output(emb)
        self._out = out.detach().clone()
        
        return out.squeeze(-1)
