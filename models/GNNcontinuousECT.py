import torch
from baseGraphModel import BaseGraphModel
from torch import nn
from utils.continuous_delta_ect import compute_continuous_delta_ect_edges
import torch_geometric
# todo get direction init for learned directions
        
import torch.nn.functional as F

from utils.utility import ECTembedMethod, ECTtype, build_neigh_batch, maxL2_graph_norm

def get_k_bfs_subgraphs(num_nodes, edge_index, k): 
    """
    num_nodes: int | obsvious;
    edge_index: 2, E | obvious;
    k: int| the output neighborhood size;
    
    expectation: size(smallest_connected_component(V,E)) <= k | behavior is undefined if not 
    """
    
    device = edge_index.device
 
    self_loops = torch.arange(num_nodes, device=device).repeat(2, 1)
    adj_edges = torch.cat([edge_index, self_loops], dim=1)
    values = torch.ones(adj_edges.size(1), device=device)
    adj = torch.sparse_coo_tensor(adj_edges, values, (num_nodes, num_nodes)).coalesce()
    #return neighborhoods, subgraph_edges

    return torch.empty(num_nodes, k), []



# Begin Credit: Nello Blaser from Encoding the Euler Characteristic by Blaser et al. 
def _responsible_counts(simplex_idx, nh_b, N_b, nd):
    """Scatter simplex attributions: each simplex goes to its max-projection node."""
    k, M = simplex_idx.shape
    h    = nh_b[simplex_idx]                                        # (k, M, nd)
    am   = h.argmax(dim=0)                                          # (M, nd)
    si_T = simplex_idx.T.unsqueeze(-1).expand(M, k, nd)            # (M, k, nd)
    resp = si_T.gather(1, am.unsqueeze(1)).squeeze(1)              # (M, nd)

    d_idx = torch.arange(nd, device=device).unsqueeze(0).expand(M, nd)
    flat  = resp * nd + d_idx
    counts = torch.zeros(N_b * nd, device=device)
    counts.scatter_add_(0, flat.flatten(), torch.ones(M * nd, device=device))
    return counts.view(N_b, nd)

def _single(nh_b, ei_b, N_b, fi_b):
    nd = nh_b.shape[1]

    ecounts = _responsible_counts(ei_b, nh_b, N_b, nd) if ei_b.shape[1] > 0 \
              else torch.zeros(N_b, nd, device=device)
    fcounts = _responsible_counts(fi_b, nh_b, N_b, nd) if fi_b is not None and fi_b.shape[1] > 0 \
              else torch.zeros(N_b, nd, device=device)

    delta = torch.ones(N_b, nd, device=device) - ecounts + fcounts  # (N_b, nd)

    # Mask: keep only nodes with non-zero ECT change in at least one direction
    keep    = (delta != 0).any(dim=1)                               # (N_b,)
    nh_k    = nh_b[keep]
    delta_k = delta[keep]

    result = torch.stack([nh_k, delta_k])                           # (2, N_keep, nd)

    if sort:
        _, sidx = nh_k.sort(dim=0)
        result = result.gather(1, sidx.unsqueeze(0).expand_as(result))
    return result
# End Credit



class lcECTLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(lcECTLayer, self).__init__() # add pe size to num_layers???
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp_project = nn.Linear(2, out_channels)
        self.include_dir_mlp = nn.Linear(out_channels + in_channels, out_channels)
  
    def forward(self, x, v, e_index, bfs_index, edge_sets):
        # x: num_nodes, D | ;
        # v : D, num_dirs | ;
        # e_index: 2, E | torch geometric;
        # bfs_index: num_nodes, k |;
        # edge_subsets: List[tensor] | for all subset in edge_subsets, subset.shape==2,E_subset, where E_subset <= E
        
        # new x 
        bfs_batches = torch.arange(bfs_index.shape[0],device=bfs_index)[:, None].broadcast_to(bfs_index.shape)
        bfs_batches_flat = bfs_batches.flatten() # num_nodes * k
        bfs_index_flat = bfs_index.flatten()

        # E, num_nodes, k
        
        bfs_edge_index = torch.gather(e_index, 1, bfs_edge_to_node_index[0][None,:].expand(2,-1)) # bfs_edge_index[i,j] = edge_index[i][bfs_index_edge_to_node_index[i][j]] # now this is all valid edges, now we just need to make it respect the order of x_bfs

        

           
        bfs_x = torch.gather(x, 0, bfs_index_flat[:, None].expand(-1, x.shape[1])) # bfs_x[i, j] = x[bfs_index[i][j]][j] # num_nodes * k, D
         
        """Continuous delta ECT for graphs.
        # TODO: at some point I may need to abandon the use of this and create my own continuous ECT function. For now I just want the proof of concept to work

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
        list_ect_reps = compute_continuous_delta_ect_edges(bfs_x, flattened_bfs, v, index=bfs_batches)
        # add F.pad below # TODO remove F.pad and make pad work with above function
        max_dim = max() 
        padded_list_ect_reps = [F.pad(ect_reps, (0,0, 0,max_dim-ect_reps.shape[1], 0,0), mode="constant", values=0.0) for ect_reps in list_ect_reps]

        padded_ect_reps = torch.stack(padded_list_ect_reps, dim=0)
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
		conv_name="GCNConv",
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
        self.v = torch.tensor(directions_init(), requires_grad=self.learn_directions)

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

