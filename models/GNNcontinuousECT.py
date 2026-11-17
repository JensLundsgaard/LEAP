import torch
from baseGraphModel import BaseGraphModel
from torch import nn
from utils.continuous_delta_ect import compute_continuous_delta_ect_edges
import torch_geometric
# todo get direction init for learned directions
dect.directions.generate_uniform_directions(
            num_thetas=config.num_thetas,
            d=dim if not config.ect_on_pe else config.pos_enc_embed_dim,
            seed=config.ect_seed,
            device=accelerator,
        )

        
import torch.nn.functional as F

from utils.utility import ECTembedMethod, ECTtype, build_neigh_batch, maxL2_graph_norm

@torch.no_grad() # no gradient is induced here, all operations are discrete
def get_k_bfs_subgraphs(num_nodes, edge_index, k, rng=torch.Generator().manual_seed(42)): 
    """
    num_nodes: int | obsvious;
    edge_index: 2, E | obvious;
    k: int| the output neighborhood size;
    
    expectation: size(smallest_connected_component(V,E)) <= k | behavior is undefined if not 

    returns:
        neighborhoods: num_nodes, K
        subgraph_edges: list | len = num_nodes, each edge_index tensor index the respective tensor in neighborhoods        
    """
    
    device = edge_index.device
 
    self_loops = torch.arange(num_nodes, device=device)[None,:].repeat(2, 1)
    adj_edges = torch.cat([edge_index, self_loops], dim=1)
    values = torch.ones(adj_edges.size(1), device=device)
    adj = torch.sparse_coo_tensor(adj_edges, values, (num_nodes, num_nodes)).coalesce()
    visited = torch.eye(num_nodes, dtype=torch.bool, device=device)
    # save distances, set dist to max distances for later topk
    hop_dist = torch.full((num_nodes, num_nodes), k, dtype=torch.long, device=device)
    hop_dist.fill_diagonal_(0)
    frontier = visited.float()
    for i in range(k):
        reach = torch.sparse.mm(adj, frontier.t()).t() > 0
        newly_reached = reach & ~visited
        if not newly_reached.any():
            break
        hop_dist[newly_reached] = i+1
        visited |= newly_reached 
        frontier = visited.float()
    hop_dist = hop_dist.float() + torch.randn(size=hop_dist.shape, device=device, generator=rng) * 0.1
        
    neighborhoods = torch.topk(hop_dist, k, dim=1, largest=False).indices # topk not being determistic is good, if we want to be truely random we can add a sufficently small random matrix to break ties randomly
    # produce each mask 
    edge_mask = ((neighborhoods[: ,None, :] == edge_index[0][None, :, None]).any(dim=2) & (neighborhoods[:, None, :] == edge_index[1][None, :, None]).any(dim=2))
    subgraph_edges = [edge_index[:, edge_mask[i]] for i in range(num_nodes)]

    # new reindex so that each edge_index actually indexes it's nbh rather than the original [num_nodes]
    # we can use a sparse tensor of neighborhoods to build an index, which we gather onto the edge index so that are e_indices index the right thing`
    values = torch.arange(k, device=device)[None, :].expand(num_nodes, -1)
    dim0_index = torch.arange(num_nodes, device=device)[:, None].expand(-1, k)
    nbh_indices = torch.stack([dim0_index, neighborhoods],dim=-1).reshape(-1, 2).T
    nbh_index_sparse = torch.sparse_coo_tensor(nbh_indices, values.flatten(), (num_nodes, num_nodes), device=device).coalesce().to_dense() # since spatial is already O(num_nodes**2) we can simply make this dense
    
    subgraph_edges = [torch.gather(nbh_index_sparse[i][None,:].expand(2,-1),1,e_i) for i,e_i in enumerate(subgraph_edges)] 
    
    return neighborhoods, subgraph_edges

   




class lcECTLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(lcECTLayer, self).__init__() # add pe size to num_layers???
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp_project = nn.Sequential(nn.Linear(2, out_channels), nn.ReLU)
        self.include_dir_mlp = nn.Linear(out_channels + in_channels, out_channels)
        self.keys = nn.Linear(out_channels, out_channels)
        self.queries = nn.Linear(out_channels, out_channels)
        self.values = nn.Linear(out_channels, out_channels)
  
    def forward(self, x, v, e_index, bfs_index, subgraph_e_indices):
        # x: num_nodes, D | ;
        # v : D, num_dirs | ;
        # e_index: 2, E | torch geometric;
        # bfs_index: num_nodes, k |;
        # subgraph_edge_indices: List[tensor] | for all subset in subgraph_edge_indices, subset.shape==2,E_subset, where E_subset <= E where the actual values index the positions of nodes in bfs_index
        
        # new x 
        bfs_batches = torch.arange(bfs_index.shape[0],device=bfs_index)[:, None].broadcast_to(bfs_index.shape)
        bfs_batches_flat = bfs_batches.flatten() # num_nodes * k
        bfs_index_flat = bfs_index.flatten()

        # need a partial injective function from [num_nodes]^2 \to [num_bfs_nodes], defined at (n_1, n_2) if n_2 in bfs_index[n_1]
        bfs_index_to_all_bfs_node = bfs_index + bfs_batches * bfs_index.shape[1] # 
        bfs_e_index = torch.cat([torch.gather(bfs_index_to_all_bfs_node[i][None, :].expand(2, -1), 0, subgraph_e_index) for i, subgraph_e_index in enumerate(subgraph_e_indices)], dim=1)
    
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
        list_ect_reps = compute_continuous_delta_ect_edges(bfs_x, bfs_e_index, v, index=bfs_batches_flat)

        # add F.pad below # TODO: remove F.pad and make padding work with function
        N_active_sizes = torch.tensor([ect_reps.shape[1] for ect_reps in list_ect_reps], device = bfs_x.device)
        
        max_dim = N_active_sizes.max().item()
        mask = N_active_sizes[:, None] < torch.arange(max_dim, device=N_active_sizes.device)[None, :] # B, N_padded
        padded_list_ect_reps = [F.pad(ect_reps, (0,0, 0,max_dim-ect_reps.shape[1], 0,0), mode="constant", values=0.0) for ect_reps in list_ect_reps]

        padded_ect_reps = torch.stack(padded_list_ect_reps, dim=0)
        # then we need to turn (num_nodes, 2, N_padded_active, num_dirs) into num_nodes, out_channels
        padded_ect_reps = padded_ect_reps.movedim(1,3)
        padded_ect_reps = padded_ect_reps.move_dim(1,2)
        tokens = self.mlp_project(padded_ect_reps)

        # TODO: move this out at some point 
        mask_attended = F.scaled_dot_product_attention(self.queries(padded_ect_reps), self.keys(padded_ect_reps), self.values(padded_ect_reps), attn_mask=mask[:,None,:,None] & mask[:, None, None, :]) # num_nodes,N_padded, num_dirs, out_channels
        
        x_pe = (mask_attended * mask[:,:,None,None]).exp().sum(dim=1).log() # num_nodes, num_dirs, out_channels
        x_pe_dirs = torch.cat([x_pe, v[None,:,:].expand(x_pe.shape[0],-1,-1)])
        
        x_pe_dir_aware = self.include_dir_mlp(x_pe_dirs) # num_nodes, num_dirs, out_channels
        
        x_pe = x_pe_dir_aware.exp().sum(dim=1).log()
            
        return x_pe
         
   

class GNNclECT(nn.Module):
    """
    For now this module descends from nn.Module, but I may generalize the BaseGraphModel to work with clECT
    This class will be able to use various different underlying MPNN's and multiple different methods for summarizing local neighborhoods with continuous ECT encodings (Blaser et al.)
    I have modified the BaseLineModel to also accept this model. It will work with the rest of the pipeline, and should be as easy as changing one argument to train_gcn, which will 
    I will try to take a look at the other wrapper since I'm still not sure what the whole flow of information, for now I am just trying to make a proof of concept work
    - Jens Lundsgaard
    """
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers, conv_name="GCNConv", **kwargs):

        super(GNNclECT, self).__init__() # add pe size to num_layers???
        for arg, val in kwargs.items():
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

        self.out_mlp = nn.Sequential(nn.Linear(out_channels, out_channels), nn.ReLU(), nn.Linear(out_channels, out_channels)) # classify

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

