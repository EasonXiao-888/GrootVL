import math
from functools import partial
from typing import Optional, Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops import rearrange, repeat


from torch.autograd import Function
from torch.autograd.function import once_differentiable
from tree_scan import _C
from .tree_scan_utils.tree_scan_core import MinimumSpanningTree

class _BFS(Function):
    @staticmethod
    def forward(ctx, edge_index, max_adj_per_vertex):
        sorted_index, sorted_parent, sorted_child =\
                _C.bfs_forward(edge_index, max_adj_per_vertex)
        return sorted_index, sorted_parent, sorted_child

class _Refine(Function):
    @staticmethod
    def forward(ctx, feature_in, edge_weight, sorted_index, sorted_parent, sorted_child,edge_coef):
        feature_aggr, feature_aggr_up, =\
            _C.tree_scan_refine_forward(feature_in, edge_weight, sorted_index, sorted_parent, sorted_child,edge_coef)
            
        ctx.save_for_backward(feature_in, edge_weight, sorted_index, sorted_parent,
                sorted_child, feature_aggr, feature_aggr_up, edge_coef)
        return feature_aggr
        # return feature_aggr_up

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        feature_in, edge_weight, sorted_index, sorted_parent,\
        sorted_child, feature_aggr, feature_aggr_up,edge_coef = ctx.saved_tensors

        grad_feature = _C.tree_scan_refine_backward_feature(feature_in, edge_weight,
                sorted_index, sorted_parent, sorted_child, feature_aggr, feature_aggr_up,
                grad_output,edge_coef)
        grad_edge_weight = _C.tree_scan_refine_backward_edge_weight(feature_in, edge_weight, 
                sorted_index, sorted_parent, sorted_child, feature_aggr, feature_aggr_up,
                grad_output,edge_coef)
        return grad_feature, grad_edge_weight, None, None, None, None

def batch_index_opr(data, index):
    with torch.no_grad():
        channel = data.shape[1]
        index = index.unsqueeze(1).expand(-1, channel, -1).long()
    data = torch.gather(data, 2, index)
    return data

def tree_scanning_core(xs, dts, 
        As, Bs, Cs, Ds,
        delta_bias,origin_shape,h_norm):

    K = 1
    _,_,H,W = origin_shape
    B, D, L = xs.shape
    dts = F.softplus(dts + delta_bias.unsqueeze(0).unsqueeze(-1))  
    # import pdb;pdb.set_trace()
    deltaA = (dts * As.unsqueeze(0)).exp_()  # b d l
    deltaB = rearrange(dts,'b (k d) l -> b k d l',k=K,d=int(D/K)) * Bs # b 1 d L
    BX = deltaB * rearrange(xs,'b (k d) l -> b k d l',k=K,d=int(D/K)) # b 1 d L

    bfs = _BFS.apply
    refine = _Refine.apply

    feat_in = BX.view(B,-1,L)   # b D L
    edge_weight = deltaA  # b D L    

    def edge_transform(edge_weight, sorted_index, sorted_child):
        edge_weight = batch_index_opr(edge_weight, sorted_index)   # b d l
        return edge_weight,

    fea4tree_hw = rearrange(xs,'b d (h w) -> b d h w',h=H,w=W)  # B d L
    mst_layer = MinimumSpanningTree("Cosine", torch.exp)
    tree = mst_layer(fea4tree_hw)
    sorted_index ,sorted_parent,sorted_child = bfs(tree,4)
    edge_weight, = edge_transform(edge_weight,sorted_index,sorted_child)
    # import pdb;pdb.set_trace()
    edge_weight_coef = torch.ones_like(sorted_index,dtype=edge_weight.dtype)  # edge coef, default by 1
    feature_out = refine(feat_in, edge_weight, sorted_index, sorted_parent, sorted_child, edge_weight_coef)

    if h_norm is not None:
        out = h_norm(feature_out.transpose(-1,-2).contiguous())

    y = (rearrange(out,'b l (k d) -> b l k d',k=K,d=int(D/K)).unsqueeze(-1) @ rearrange(Cs,'b k n l -> b l k n').unsqueeze(-1)).squeeze(-1) # (B L K D N) @ (B L K N 1) -> (B L K D 1)
    # import pdb;pdb.set_trace()
    y = rearrange(y,'b l k d -> b (k d) l')
    y = y + Ds.reshape(1,-1,1) * xs
    return y

def tree_scanning(
    x: torch.Tensor=None, 
    x_proj_weight: torch.Tensor=None,
    x_proj_bias: torch.Tensor=None,
    dt_projs_weight: torch.Tensor=None,
    dt_projs_bias: torch.Tensor=None,
    A_logs: torch.Tensor=None,
    Ds: torch.Tensor=None,
    out_norm: torch.nn.Module=None,
    to_dtype=True,
    force_fp32=False, # False if ssoflex
    h_norm=None,
):

    B, D, H, W = x.shape
    origin_shape = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W
    
    xs = rearrange(x.unsqueeze(1),'b k d h w -> b k d (h w)') 
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    As = -torch.exp(A_logs.to(torch.float)) # (c, d)
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float) # (c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    force_fp32 = True
    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    ys = tree_scanning_core(xs, dts,
                           As, Bs, Cs, Ds,
                             delta_bias,origin_shape,h_norm).view(B,K,-1,H,W)

    y = rearrange(ys,'b k d h w -> b (k d) (h w)')
    y = y.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
    y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)

class Tree_SSM(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        ssm_rank_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        **kwargs,
    ):
        """
        ssm_rank_ratio would be used in the future...
        """
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state 
        self.d_conv = d_conv

        self.out_norm = nn.LayerNorm(d_inner)
        self.h_norm = nn.LayerNorm(d_inner)

        self.K = 1
        self.K2 = self.K

        # in proj =======================================
        d_proj = d_expand * 2
        self.in_proj = nn.Linear(d_model, d_proj, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()
        
        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj
        
        # out proj =======================================
        self.out_proj = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs
        
        # A, D =======================================
        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True) # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True) # (K * D)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D


    def forward_core(self, x: torch.Tensor, channel_first=False, force_fp32=None):
        force_fp32 = self.training if force_fp32 is None else force_fp32
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        x = tree_scanning(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, 
            out_norm=getattr(self, "out_norm", None),
            force_fp32=force_fp32, h_norm=self.h_norm,
        )
        return x
    
    def forward(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1) # (b, h, w, d)
        z = self.act(z)
        if self.d_conv > 0:
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.conv2d(x) # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x, channel_first=(self.d_conv > 1))
        y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
