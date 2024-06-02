import torch
import torch.distributed as dist

from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from einops import rearrange

from tree_scan import _C

class _MST(Function):
    @staticmethod
    def forward(ctx, edge_index, edge_weight, vertex_index):
        edge_out = _C.mst_forward(edge_index, edge_weight, vertex_index)
        return edge_out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        return None, None, None

mst = _MST.apply

def norm2_distance(fm_ref, fm_tar):
    diff = fm_ref - fm_tar
    weight = (diff * diff).sum(dim=1)
    return weight

def norm1_distance(fm_ref, fm_tar):
    diff = fm_ref - fm_tar
    weight = torch.abs(diff).sum(dim=1)
    return weight

class MinimumSpanningTree(nn.Module):
    def __init__(self, distance_func, mapping_func=None):
        super(MinimumSpanningTree, self).__init__()
        self.distance_func = distance_func
        self.mapping_func = mapping_func
    
    @staticmethod
    def _build_matrix_index(fm):
        batch, height, width = (fm.shape[0], *fm.shape[2:])
        row = torch.arange(width, dtype=torch.int32, device=fm.device).unsqueeze(0)
        col = torch.arange(height, dtype=torch.int32, device=fm.device).unsqueeze(1)
        raw_index = row + col * width
        row_index = torch.stack([raw_index[:-1, :], raw_index[1:, :]], 2)
        col_index = torch.stack([raw_index[:, :-1], raw_index[:, 1:]], 2)
        index = torch.cat([row_index.reshape(1, -1, 2), 
                           col_index.reshape(1, -1, 2)], 1)
        index = index.expand(batch, -1, -1)
        return index

    def _build_feature_weight(self, fm):
        batch = fm.shape[0]
        weight_row = norm2_distance(fm[:, :, :-1, :], fm[:, :, 1:, :])
        weight_col = norm2_distance(fm[:, :, :, :-1], fm[:, :, :, 1:])
        weight_row = weight_row.reshape([batch, -1])
        weight_col = weight_col.reshape([batch, -1])
        weight = torch.cat([weight_row, weight_col], dim=1)
        if self.mapping_func is not None:
            weight = self.mapping_func(weight)
        return weight

    def _build_feature_weight_cosine(self, fm, max_tree):
        batch,dim = fm.shape[0],fm.shape[1]
        weight_row = torch.cosine_similarity(fm[:, :, :-1, :].reshape(batch,dim,-1), fm[:, :, 1:, :].reshape(batch,dim,-1),dim=1)
        # import pdb;pdb.set_trace()
        weight_col = torch.cosine_similarity(fm[:, :, :, :-1].reshape(batch,dim,-1), fm[:, :, :, 1:].reshape(batch,dim,-1),dim=1)
        weight = torch.cat([weight_row, weight_col], dim=1)
        if self.mapping_func is not None:
            if max_tree:
                weight = self.mapping_func(weight)   # cosine similarity needs "-weight" for min tree, "weight" for max tree
            else:
                weight = self.mapping_func(-weight)   # cosine similarity needs "-weight" for min tree, "weight" for max tree
        return weight

    def forward(self, guide_in, max_tree=False):
        with torch.no_grad():
            index = self._build_matrix_index(guide_in)
            if self.distance_func == "Cosine":
                weight = self._build_feature_weight_cosine(guide_in,max_tree)
            else:
                weight = self._build_feature_weight(guide_in)
            tree = mst(index, weight, guide_in.shape[2] * guide_in.shape[3])
            # tree = mst(index, weight, guide_in.shape[2])
        return tree
