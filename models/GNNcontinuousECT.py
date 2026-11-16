import torch
from baseGraphModel import BaseGraphModel
from torch import nn
from utils.continuous_delta_ect import compute_continuous_delta_ect_edges
import torch_geometric
# todo get direction init for learned directions
        
import torch.nn.functional as F



 class lcECTLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(lcECTLayer, self).__init__() # add pe size to num_layers???
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp_project = nn.Linear(2, out_channels)
        self.include_dir_mlp = nn.Linear(out_channels + in_channels, out_channels)
  
    def forward(self, x, v, e_index, bfs_index):
        # x: num_nodes, D | ;
        # v : D, num_dirs | ;
        # e_index: 2, E | torch geometric;
        # bfs_index: num_nodes, k |;
        bfs_batches = torch.arange(bfs_index.shape[0],device=bfs_index)[:, None].broadcast_to(bfs_index.shape)
        bfs_batches = bfs_batches.flatten()
        flattened_bfs = bfs_index.flatten()
         
        """Continuous delta ECT for graphs.

        Args:
            x:          Node feature matrix [N, D].
            edge_index: Edge index [2, E] in torch-geometric format.
            v:          Direction matrix [D, num_dirs].
            faces:      Optional face index tensor [3, F] for triangle meshes.
            index:      Optional batch vector [N].
            sort:       If True, sort nodes by ascending threshold per direction.

        Returns:
            [2, N_active, num_dirs] (single graph) or list of [2, N_b_active, num_dirs] (batch).
            Row 0: filtration threshold (projection of the node)
            Row 1: ECT delta (delta = #vertices - #edges + #faces attributed to this node)
            Nodes with delta == 0 in all directions are masked out.
        """
        list_ect_reps = compute_continuous_delta_ect_edges(x, flattened_bfs, v, index=bfs_batches)
        # add F.pad below # TODO remove F.pad and make pad work with above function
        padded_list_ect_reps = F.pad()
        # then we need to turn (num_nodes, 2, N_padded_active, num_dirs) into num_nodes, out_channels
        masked_attended = attend(self.keys(padded_ect_reps), self.queries(padded_ect_reps), self.values(padded_ect_reps), mask=mask) # num_nodes,N_padded, num_dirs, out_channels
        x_pe = (mask_attended * mask).log().sum(dim=1).exp() # num_nodes, num_dirs, out_channels
        
        x_pe 
        return x_pe
         
   

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
        k = 16,
		conv_name="GCNConv",
        learn_directions=True,
		**kwargs):
        super(GNNclECT, self).__init__() # add pe size to num_layers???
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
		self.mlp_in = self.Sequential(self.Linear(self.num_feat_1st_layer, self.hidden_dim - self.pe_dim), nn.ReLU()) # project
		# Graph Conv Layers
		self.convs.append(Conv(hidden_channels, hidden_channels - self.pe_dim))
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
            self.pe_encoders.append(lcECTLayer(self.hidden_dim-self.pe_dim, self.pe_dim))
            
        
        
        self.out_mlp = nn.Sequential(nn.Linear(out_channels, out_channels), nn.ReLU, nn.Linear(out_channels, out_channels)) # classify

        self.num_layers = num_layers 
        self.k = k 

	def forward(self, data, **kwargs):
        x, e_index = data.x, data.edge_index
        
        bfs_e_index = self._build_bfs_nbhds(e_index, k=self.k) # this could also happen in the dataloader for efficiency for now it can stay though
        x = self.mpl_in(x) 

        for i in range(self.num_layers -1):
            x_pe = self.pe_encoders[i](x, e_index, bfs_e_index)
            x = torch.cat([x, x_pe], dim=-1)
            x = self.mlps[i](x)
            x = self.convs[i](x, e_index)

        x_pe = self.pe_encoders[-1](x, e_index, bfs_e_index)
        x = torch.cat([x, x_pe], dim=-1)
        x = self.convs[-1](x, e_index)
        x = self.out_mlp(x)
        return x
	
    def _build_bfs_nbhds(self, e_index, k=16):
         
        return 

