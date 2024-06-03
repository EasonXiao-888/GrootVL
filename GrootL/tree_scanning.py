from torch.autograd import Function
from torch.autograd.function import once_differentiable
from tree_scan_lan import _C
import torch
import torch.nn as nn
from einops import rearrange,repeat

class _MST(Function):
    @staticmethod
    def forward(ctx, edge_index, edge_weight, vertex_index):
        edge_out = _C.mst_forward(edge_index, edge_weight, vertex_index)
        return edge_out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        return None, None, None

class _BFS(Function):
    @staticmethod
    def forward(ctx, edge_index, max_adj_per_vertex):
        sorted_index, sorted_parent, sorted_child,_ =\
                _C.bfs_forward(edge_index, max_adj_per_vertex)
        return sorted_index, sorted_parent, sorted_child

class _Refine(Function):
    @staticmethod
    def forward(ctx, feature_in, edge_weight, sorted_index, sorted_parent, sorted_child):
        feature_out =\
            _C.tree_scan_refine_forward(feature_in, edge_weight, sorted_index, sorted_parent, sorted_child)
            
        ctx.save_for_backward(feature_out, edge_weight, sorted_index, sorted_parent,
                sorted_child)
        return feature_out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        feature_out, edge_weight, sorted_index, sorted_parent,\
        sorted_child = ctx.saved_tensors

        grad_feature, grad_edge = _C.tree_scan_refine_backward_feature(feature_out, edge_weight
                ,sorted_index, sorted_parent, sorted_child,
                grad_output)
        return grad_feature, grad_edge, None, None, None

def norm2_distance(fm_ref, fm_tar):
    diff = fm_ref - fm_tar
    weight = (diff * diff).sum(dim=-2)
    return torch.exp(weight)     # with - is for max tree

def cosine_distance(fm_ref, fm_tar):
    weight = -torch.cosine_similarity(fm_ref, fm_tar,dim=1)
    return torch.exp(weight)    #with - is for min tree

def batch_index_opr(data, index):
    with torch.no_grad():
        channel = data.shape[1]
        index = index.unsqueeze(1).expand(-1, channel, -1).long()
    data = torch.gather(data, 2, index)
    return data

def tree_scanning_algorithm(self, input_states, contex_len, cache_params):
    batch_size, seq_len, _ = input_states.shape
    dtype = input_states.dtype
    device = input_states.device
    # 1. Gated MLP's linear projection
    projected_states = self.in_proj(input_states).transpose(1, 2)                   # [batch, 2 * intermediate_size, seq_len]
    hidden_states, gate = projected_states.chunk(2, dim=1)

    hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])         # [batch, intermediate_size, seq_len]
    # 3. State Space Model sequence transformation
    # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
    ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
    time_step, B, C = torch.split(
        ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
    )
    discrete_time_step = self.dt_proj(time_step)                                    # [batch, seq_len, intermediate_size]
    discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(1, 2) # [batch, intermediate_size, seq_len]
    # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
    A = -torch.exp(self.A_log.float())                                              # [intermediate_size, ssm_state_size]
    discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None]) # [batch, intermediate_size, seq_len, ssm_state_size]
    discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()       # [batch, intermediade_size, seq_len, ssm_state_size]
    deltaB_u = discrete_B * hidden_states[:, :, :, None].float()
    ### tree scan
    weight = rearrange(discrete_A,'b d l n -> b (d n) l').contiguous()
    feature_in = rearrange(deltaB_u,'b d l n -> b (d n) l').contiguous()
    feature_in = torch.flip(feature_in,dims=[-1]).contiguous()
    weight = torch.roll(torch.flip(weight,dims=[-1]),1,-1).contiguous()

    mst = _MST.apply
    bfs = _BFS.apply
    refine = _Refine.apply

    ### hand-build tree
    tree_ = []
    for i in range(seq_len-1):
        tree_.append([i, i + 1])
    tree_ = torch.tensor(tree_,dtype=torch.int32).to(device)
    tree = tree_.repeat(batch_size,1,1)
    sorted_index1 ,sorted_parent1,sorted_child1 = bfs(tree,4)
    
    ### build tree by feature
    try:
        contex_len = min(contex_len)
    except:
        contex_len = contex_len
    with torch.no_grad():
        def generate_pairs(L,prompt_len):
            pairs = []
            for i in range(0, L-prompt_len):
                pairs.append([i,i+1])
            for i in range(L-prompt_len,L-3):
                pairs.append([i, i+1])
                pairs.append([i, i+2])
                pairs.append([i, i+3])
            pairs.append([L-3, L-2])
            pairs.append([L-3, L-1])
            pairs.append([L-2, L-1])
            return pairs
        # import pdb;pdb.set_trace()
        if contex_len > 2:
            pairs = torch.tensor(generate_pairs(seq_len,contex_len),dtype=torch.int32,device=feature_in.device)
            data1 = torch.index_select(feature_in,2,pairs[:,0])
            data2 = torch.index_select(feature_in,2,pairs[:,1])
            # import pdb;pdb.set_trace()
            tree_weight = cosine_distance(data1,data2)

            tree = mst(pairs.repeat(batch_size,1,1),tree_weight,seq_len)
            sorted_index2, sorted_parent2, sorted_child2 = bfs(tree,contex_len)
        else:
            sorted_index2 ,sorted_parent2, sorted_child2 = sorted_index1 ,sorted_parent1,sorted_child1

        # import pdb;pdb.set_trace()
    # import pdb;pdb.set_trace()
    feature_out1 = refine(feature_in, weight, sorted_index1, sorted_parent1, sorted_child1)
    # import pdb;pdb.set_trace()
    edge_weight = batch_index_opr(weight, sorted_index2)
    feature_out2 = refine(feature_in, edge_weight, sorted_index2, sorted_parent2, sorted_child2)
    feature_out  = feature_out2 * 0.3 + feature_out1  # 0.3 is scaling factor (hyperparameter)

    feature_out = rearrange(torch.flip(feature_out.to(dtype),dims=[-1]),'b (d n) l -> b l d n',b=batch_size,n=discrete_A.shape[-1]).contiguous()
    scan_output_ = (feature_out @ C.unsqueeze(-1)).squeeze(-1).transpose(-1,-2) # (B, L, D, N) @ (B, L, N, 1) -> (B, L, D, 1)
    
    # [batch, seq_len, intermediade_size]
    scan_output = scan_output_ + (hidden_states * self.D[None, :, None])
    scan_output = (scan_output * self.act(gate))
    # 4. Final linear projection
    contextualized_states = self.out_proj(scan_output.transpose(1, 2))             # [batch, seq_len, hidden_size]
    return contextualized_states


def slow_forward(self,input_states, cache_params):
    batch_size, seq_len, _ = input_states.shape
    dtype = input_states.dtype
    # 1. Gated MLP's linear projection
    projected_states = self.in_proj(input_states).transpose(1, 2)                   # [batch, 2 * intermediate_size, seq_len]
    hidden_states, gate = projected_states.chunk(2, dim=1)
    # 2. Convolution sequence transformation
    if cache_params is not None:
        ssm_state = cache_params.ssm_states[self.layer_idx]
        if cache_params.seqlen_offset > 0:
            conv_state = cache_params.conv_states[self.layer_idx]                   # [batch, intermediate_size, conv_kernel_size]
            conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
            conv_state[:, :, -1] = hidden_states[:, :, 0]
            cache_params.conv_states[self.layer_idx].copy_(conv_state)
            hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
            if self.use_conv_bias:
                hidden_states += self.conv1d.bias
            hidden_states = self.act(hidden_states).to(dtype).unsqueeze(-1)         # [batch, intermediate_size, 1] : decoding
        else:
            conv_state = nn.functional.pad(
                hidden_states,
                (self.conv_kernel_size - hidden_states.shape[-1], 0)
            )
            cache_params.conv_states[self.layer_idx].copy_(conv_state)
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])     # [batch, intermediate_size, seq_len]
    else:
        ssm_state = torch.zeros(
            (batch_size, self.intermediate_size, self.ssm_state_size),
            device=hidden_states.device, dtype=dtype
        )
        hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])         # [batch, intermediate_size, seq_len]
    # 3. State Space Model sequence transformation
    # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
    ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
    time_step, B, C = torch.split(
        ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
    )
    discrete_time_step = self.dt_proj(time_step)                                    # [batch, seq_len, intermediate_size]
    discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(1, 2) # [batch, intermediate_size, seq_len]
    # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
    A = -torch.exp(self.A_log.float())                                              # [intermediate_size, ssm_state_size]
    discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None]) # [batch, intermediate_size, seq_len, ssm_state_size]
    discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()       # [batch, intermediade_size, seq_len, ssm_state_size]
    deltaB_u = discrete_B * hidden_states[:, :, :, None].float()
    # 3.c perform the recurrence y ← SSM(A, B, C)(x)
    scan_outputs = []
    for i in range(seq_len):
        ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]      # [batch, intermediade_size, ssm_state]
        scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))  # [batch, intermediade_size, 1]
        scan_outputs.append(scan_output[:, :, 0])
    scan_output = torch.stack(scan_outputs, dim=-1)   
    # return scan_output                    # [batch, seq_len, intermediade_size]
    scan_output = scan_output + (hidden_states * self.D[None, :, None])
    scan_output = (scan_output * self.act(gate))
    if cache_params is not None:
        cache_params.ssm_states[self.layer_idx].copy_(ssm_state)
    # 4. Final linear projection
    contextualized_states = self.out_proj(scan_output.transpose(1, 2))             # [batch, seq_len, hidden_size]
    return contextualized_states