#include <math.h>
#include <thread>
#include <vector>
#include <deque>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <THC/THCAtomics.cuh>
#include <ATen/native/cuda/KernelUtils.cuh>

#define CUDA_NUM_THREADS         64
#define GET_CUDA_CHANNEL(N)      ceil(512.0f / N)

template <typename scalar_t>
__global__ void leaf_root_aggr_kernel_template(
        scalar_t *in_data, 
        scalar_t *out_data, 
        scalar_t *weight,
        int *sorted_index, 
        int *sorted_child_index, 
        int batch_size, 
        int channel_size, 
        int vertex_count,
        int max_adj_per_node){

    const int thread_idx    = threadIdx.x;
    const int batch_idx     = blockIdx.x;
    const int channel_idx   = blockIdx.y;
    const int thread_count  = blockDim.x;
    const int channel_step  = gridDim.y;
    
    if (in_data != NULL){
        in_data    += batch_idx * vertex_count * channel_size;
    }    
    out_data             += batch_idx * vertex_count * channel_size;
    weight               += batch_idx * vertex_count * channel_size;
    sorted_index         += batch_idx * vertex_count;
    sorted_child_index   += batch_idx * vertex_count * max_adj_per_node;

    __shared__ int node_per_thread[CUDA_NUM_THREADS];
    node_per_thread[thread_idx] = vertex_count;
    __syncthreads();

    int i = vertex_count - thread_idx - 1;
    while (i >= 0){
        int child_len = 0;
        bool valid = true;
        for (int j = 0; j < max_adj_per_node; j++){
            int child        = sorted_child_index[i * max_adj_per_node + j];
            int child_thread = (vertex_count - child - 1) % thread_count;

            if (child <= 0) break;
            if (node_per_thread[child_thread] > child){
                valid = false;
                break;
            }
            child_len++;
        }
        if (valid){
            int cur_pos = sorted_index[i];
            for (int k = channel_idx * vertex_count; k < channel_size * vertex_count; 
                    k += channel_step * vertex_count){
                scalar_t aggr_sum;
                if (in_data != NULL)    
                    aggr_sum = in_data[cur_pos + k];
                else
                    aggr_sum = 1;
                for (int j = 0; j < child_len; j++){
                    int child = sorted_child_index[i * max_adj_per_node + j];
                    int child_pos = sorted_index[child];
                    // int weight_offset = child * channel_size + channel_idx;
                    // aggr_sum += out_data[child + k] * weight[weight_offset];
                    aggr_sum += out_data[child_pos + k] * weight[child + k];
                }
                out_data[cur_pos + k] = aggr_sum;
            }
            node_per_thread[thread_idx] = i;
            i -= thread_count;
        }
        __syncthreads();
    }
}

template <typename scalar_t>
__global__ void root_leaf_grad_aggre_kernel_template(
        scalar_t *out_fea, 
        scalar_t *grad_in, 
        scalar_t *out_data, 
        scalar_t *out_edge_data, 
        scalar_t *weight,
        int *sorted_index, 
        int *sorted_parent_index, 
        int batch_size, 
        int channel_size, 
        int vertex_count){

    const int thread_idx    = threadIdx.x;
    const int batch_idx     = blockIdx.x;
    const int channel_idx   = blockIdx.y;
    const int thread_count  = blockDim.x;
    const int channel_step  = gridDim.y;

    out_fea             += batch_idx * vertex_count * channel_size;
    grad_in             += batch_idx * vertex_count * channel_size;
    out_data            += batch_idx * vertex_count * channel_size;
    out_edge_data       += batch_idx * vertex_count * channel_size;
    weight              += batch_idx * vertex_count * channel_size;
    sorted_index        += batch_idx * vertex_count;
    sorted_parent_index += batch_idx * vertex_count;

    __shared__ int node_per_thread[CUDA_NUM_THREADS];
    node_per_thread[thread_idx] = -1;
    // if (thread_idx == 0){
    //     // Initialize the first element of weight and sorted_parent_index
    //     for (int c = 0; c < channel_size; ++c) {
    //         weight[c * vertex_count] = 0;
    //         sorted_parent_index[c * vertex_count] = 0;
    //     }
    // }
    // __syncthreads();

    int i = thread_idx;
    while (i < vertex_count){
        int par = sorted_parent_index[i];
        int par_thread = par % thread_count;
        if ((node_per_thread[par_thread] >= par) || (i == 0)){
            int cur_pos = sorted_index[i];
            int par_pos = sorted_index[par];
            for (int k = channel_idx * vertex_count; k < channel_size * vertex_count;
                       k += channel_step * vertex_count){
                scalar_t edge_weight = weight[i + k];
                out_data[cur_pos + k] = grad_in[cur_pos + k] +
                                        out_data[par_pos + k] * edge_weight;
                if (i > 0){
                    out_edge_data[i + k] = out_data[par_pos + k] * out_fea[cur_pos + k];
                    __threadfence_block();
                }
                else
                    out_edge_data[i + k] = 0;
            }
            node_per_thread[thread_idx] = i;
            i += thread_count;
        }
        __syncthreads();
    }
}

at::Tensor tree_scan_refine_forward(
        const at::Tensor & feature_in_tensor, 
        const at::Tensor & edge_weight_tensor, 
        const at::Tensor & sorted_index_tensor, 
        const at::Tensor & sorted_parent_tensor, 
        const at::Tensor & sorted_child_tensor
    ){
    
    const int batch_size        = feature_in_tensor.size(0);
    const int channel_size      = feature_in_tensor.size(1); 
    const int vertex_size       = feature_in_tensor.size(2);
    const int max_adj_per_node  = sorted_child_tensor.size(2);

    auto options                  = feature_in_tensor.options();
    auto feature_aggr_up_tensor   = at::zeros_like(feature_in_tensor, options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 feature_block_dims(CUDA_NUM_THREADS, 1, 1), feature_grid_dims(batch_size, channel_size, 1);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(feature_in_tensor.scalar_type(), "tree_scan_refine_forward_cuda", ([&] {
        auto feature_in_data = feature_in_tensor.data_ptr<scalar_t>();
        auto edge_weight_data = edge_weight_tensor.data_ptr<scalar_t>();
        auto feature_aggr_up_data = feature_aggr_up_tensor.data_ptr<scalar_t>();
        int* sorted_index_data = sorted_index_tensor.data_ptr<int>();
        int* sorted_parent_index_data = sorted_parent_tensor.data_ptr<int>();
        int* sorted_child_index_data = sorted_child_tensor.data_ptr<int>();

        leaf_root_aggr_kernel_template<scalar_t><<< feature_grid_dims, feature_block_dims, sizeof(int) * CUDA_NUM_THREADS, stream >>>(
            feature_in_data, feature_aggr_up_data, edge_weight_data, sorted_index_data, sorted_child_index_data, batch_size, channel_size, vertex_size, max_adj_per_node);
    }));

    return feature_aggr_up_tensor;
}

std::tuple<at::Tensor, at::Tensor> tree_scan_refine_backward_feature(
        const at::Tensor & feature_in_tensor, 
        const at::Tensor & edge_weight_tensor, 
        const at::Tensor & sorted_index_tensor, 
        const at::Tensor & sorted_parent_tensor, 
        const at::Tensor & sorted_child_tensor,
        const at::Tensor & grad_out_tensor
    ){

    auto options                        = feature_in_tensor.options();
    auto grad_feature_tensor            = at::zeros_like(feature_in_tensor, options);
    auto grad_edge_tensor   = at::zeros_like(feature_in_tensor, options);

    const int batch_size        = feature_in_tensor.size(0);
    const int channel_size      = feature_in_tensor.size(1); 
    const int vertex_size       = feature_in_tensor.size(2);
    const int max_adj_per_node  = sorted_child_tensor.size(2);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 feature_block_dims(CUDA_NUM_THREADS, 1, 1), feature_grid_dims(batch_size, channel_size, 1);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(feature_in_tensor.scalar_type(), "tree_scan_refine_backward_feature_cuda", ([&] {

        scalar_t * feature_in          = feature_in_tensor.contiguous().data_ptr<scalar_t>();
        scalar_t * edge_weight         = edge_weight_tensor.contiguous().data_ptr<scalar_t>();
        int * sorted_index          = sorted_index_tensor.contiguous().data_ptr<int>();
        int * sorted_parent_index   = sorted_parent_tensor.contiguous().data_ptr<int>();
        int * sorted_child_index    = sorted_child_tensor.contiguous().data_ptr<int>();
        scalar_t * grad_out            = grad_out_tensor.contiguous().data_ptr<scalar_t>();
        scalar_t * grad_feature        = grad_feature_tensor.contiguous().data_ptr<scalar_t>();
        scalar_t * grad_edge        = grad_edge_tensor.contiguous().data_ptr<scalar_t>();

        root_leaf_grad_aggre_kernel_template<scalar_t><<< feature_grid_dims, feature_block_dims, sizeof(int) * CUDA_NUM_THREADS, stream >>>(
                feature_in, grad_out, grad_feature, grad_edge, edge_weight, sorted_index, sorted_parent_index, batch_size, channel_size,
                vertex_size);
    }));

    // grad_feature_tensor += (self_weight_tensor - 1).unsqueeze(1) * grad_out_tensor;
    auto result = std::make_tuple(grad_feature_tensor, grad_edge_tensor);

    return result;
}