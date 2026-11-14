import torch
from baseGraphModel import BaseGraphModel
from torch import nn
from utils.continuous_delta_ect import compute_continuous_delta_ect_edges
import torch_geometric


class GNNclECT(nn.Module):
    """
    For now this module descends from nn.Module, but I may generalize the BaseGraphModel to work with clECT
    This class will be able to use various different underlying MPNN's and multiple different methods for summarizing local neighborhoods with continuous ECT encodings (Blaser et al.)
    I have modified the BaseLineModel to also accept this model. It will work with the rest of the pipeline, and should be as easy as changing one argument to train_gcn, which will 
    I will try to take a look at the other wrapper since I'm still not sure what the whole flow of information, for now I am just trying to make a proof of concept work
    - Jens Lundsgaard
    """
    def __init__(self,
        in_channels,
		out_channels,
		hidden_channels,
		num_layers,
		conv_name="GCNConv",
		**kwargs):
        super(GNNclECT, self).__init__(in_channels, out_channels, hidden_channels, num_layers, **kwargs) # add pe size to num_layers???
        for arg, val in kwargs.items()
            setattr(self, arg, val)
        
        if conv_name == "GCNConv":
            Conv = torch_geometric.nn.GCNConv
        elif conv_name == "GATConv":
            Conv = torch_geometric.nn.GATConv
        elif conv_name == "GINConv": # not implemented yet
            Conv = lambda in_size, out_size : torch_geometric.nn.GINConv(nn.Sequential(nn.Linear(in_size, out_size),nn.ReLU()))
        else: 
            raise ValueError("select valid GNN") 

        self.conv_name = conv_name
    
		self.convs = nn.ModuleList()
			
		# Graph Conv Layers
		self.convs.append(Conv(self.num_feat_1st_layer, hidden_channels - self.pe_dim))
		for _ in range(num_layers - 2):
			self.convs.append(Conv(hidden_channels, hidden_channels - self.pe_dim))
		self.convs.append(Conv(hidden_channels, out_channels))

        self.mlps = nn.ModuleList()
        if self.conv_name != "GINConv": # gin conv already has an MlP
            for _ in range(num_layers-1):
                self.mlps.append(nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU()))
        else:
            for _ in range(num_layers-1):
                self.mlps.append(nn.Identity())


        self.mlps.append(nn.Identity())
        
        self.pe_encoders = nn.ModuleList()
        for _ in range(num_layers-1):
            self.pe_encoders.append(lcECTLayer(self.hidden_dim, self.pe_dim))
        
        
        self.mlp = nn.Sequential(nn.Linear(out_channels, out_channels), nn.ReLU, nn.Linear(out_channels, out_channels)) # classify

        self.num_layers = num_layers 

	def forward(self, data, **kwargs):
        x, e_index = data.x, data.edge_index


        for i in range(self.num_layers -1):
            x = self.convs[i](x, e_index) + x
            x = self.
            x = self.mlps[i]
        return x
	
    def _apply_convs_with_clECT(self, x, e_index, **kwargs):
        return x

    def _common_initial_forward(self, x, e_index, **kwargs):
        return x
