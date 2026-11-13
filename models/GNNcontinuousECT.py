import torch
from baseGraphModel import BaseGraphModel
from torch import nn
from utils.continuous_delta_ect from 

class GNNclECT(nn.Module):
    """
    For now this module descends from nn.Module, but I may generalize the BaseGraphModel to work with clECT
    """
    def __init__(self,
        in_channels,
		out_channels,
		hidden_channels,
		num_layers,
		conv_name="GCNConv",
		**kwargs):
        super(GNNclECT, self).__init__(in_channels, out_channels, hidden_channels, num_layers, **kwargs)
        Conv     
    
		self.convs = torch.nn.ModuleList()
		
		self.learnable_pe = kwargs.get("learnable_pe", False)

		if self.learnable_pe:
			self.learned_pe_dim = kwargs.get("learned_pe_dim", 5)
			self.pe_convs = torch.nn.ModuleList()
			
			# Learnable PE Layers
			self.pe_convs.append(Conv(self.num_feat_1st_layer - in_channels, self.learned_pe_dim))
			for _ in range(num_layers - 2):
				self.pe_convs.append(Conv(self.learned_pe_dim, self.learned_pe_dim))

		else:
			self.learned_pe_dim = 0
		

		# Graph Conv Layers
		self.convs.append(Conv(self.num_feat_1st_layer, hidden_channels - self.learned_pe_dim))
		for _ in range(num_layers - 2):
			self.convs.append(Conv(hidden_channels, hidden_channels - self.learned_pe_dim))
		self.convs.append(Conv(hidden_channels, out_channels))


	def forward(self, x, e_index, **kwargs):

		# Handle computation of ECT/PE, projections, concatenation etc.
		x = self._common_initial_forward(x, e_index, **kwargs)
	
		# pass through the GCN layers
		if self.learnable_pe:
			return self._apply_convs_with_clECT(x, e_index)

		return self._apply_convs(x, e_index)

    def _apply_convs_with_clECT(self, x, e_index):
        return x

    def _common_initial_forward(x, e_index, **kwargs):
        return x
