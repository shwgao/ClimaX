#include <iostream>
#include <torch/extension.h>
#include <cuda_runtime.h>

std::vector<int64_t> get_stride_by_shape(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> stride;
    for (auto iter=shape.begin()+1; iter!=shape.end(); iter++)
    {
        stride.push_back(std::accumulate(iter, shape.end(), 1, std::multiplies<int>()));
    }
    stride.push_back(1);    
    return stride;
}

std::vector<int64_t> get_stack_out_shape(const std::vector<torch::Tensor>& inputs, int dim){
    int num_input = inputs.size();
    int ndim = inputs[0].dim();
    std::vector<int64_t> out_shape;
    for (int i=0; i<ndim; i++)
    {
        out_shape.push_back(inputs[i].size(i));
    }
    out_shape.insert(out_shape.begin()+dim, num_input);
    return out_shape;
}

std::vector<int64_t> get_out_shape_by_stack_shape(const std::vector<int64_t> stack_shape, const std::vector<int64_t> index_shape, int64_t index)
{
    std::vector<int64_t> out_shape(stack_shape);
    out_shape.erase(out_shape.begin() + index);
    for(int64_t i=index_shape.size()-1; i>=0; i--)
    {
        out_shape.insert(out_shape.begin() + index,  index_shape[i]);
    }
    return out_shape;
}

template <int ndim, typename scalar_t>
struct TensorMetadata
{
public:
    int ndim_ = ndim;
    scalar_t* data_ptr ;
    int size[ndim];
    int stride[ndim];
};

template <int ndim>
struct Coord
{
public:
    Coord()
    {
        for (int i=0; i<ndim; i++)
        {
            index[i] = -1;
        }
    }
    int ndim_ = ndim;
    int64_t index[ndim];
};

template <typename scalar_t>
__device__ scalar_t device_add(scalar_t a, scalar_t b)
{
    return a+b;
}

template <>
__device__ half device_add(half a, half b)
{
    return __hadd(a, b);
}

// out_coord [u,v,w,x,y,z] -> stack_coord[u, x, y, z] -> src_coord[x, y, z]
template <int input_ndim, int index_ndim, typename in_scalar_t, typename index_scalar_t>
__global__ 
void fused_kernel(
        TensorMetadata<input_ndim+index_ndim, in_scalar_t> out_metadata,
        TensorMetadata<input_ndim+1, in_scalar_t> stack_metadata,
        TensorMetadata<index_ndim, index_scalar_t> index_metadata,
        int64_t num_input,
        TensorMetadata<input_ndim, in_scalar_t>* input_metadatas,
        Coord<input_ndim+index_ndim> out_coord,
        Coord<index_ndim> index_coord,
        Coord<input_ndim+1> stack_coord,
        Coord<input_ndim> src_coord,
        Coord<3> add_coord,
        TensorMetadata<3, in_scalar_t> add_metadata,
        int64_t stack_dim,
        int64_t index_dim,
        int64_t numel)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // int64_t debug_idx=1447;
    if( idx < numel) {
        // compute out coord
        int64_t remain_index = idx;
        for (int64_t i=0; i<input_ndim+index_ndim; i++)
        {
            int64_t current_dim_index = remain_index/out_metadata.stride[i];
            out_coord.index[i] = current_dim_index;
            remain_index -=  current_dim_index * out_metadata.stride[i];
        }

        //compute index_coord
        int64_t counter = index_dim;
        for (int64_t i=0; i<index_ndim; i++)
        {
            index_coord.index[i] = out_coord.index[counter];
            counter++;
        }

        // compute index element
        int64_t index_offset = 0;
        for (int64_t i=0; i<index_ndim; i++)
        {
            index_offset += index_coord.index[i] * index_metadata.stride[i];
        }
        int64_t index_element = index_metadata.data_ptr[index_offset];

        // compute stack coord
        for (int64_t i=0; i<input_ndim+1; i++){
            if (i<index_dim){
                stack_coord.index[i] = out_coord.index[i];
            }
            else if (i==index_dim){
                stack_coord.index[i] =index_element;
            }
            else{
                stack_coord.index[i] = out_coord.index[i+index_ndim-1];
            }
        }

        //compute src coord
        int64_t tensor_index = stack_coord.index[stack_dim];
        for (int i=0; i<input_ndim; i++)
        {
            if (i<stack_dim)
            {
                src_coord.index[i] = stack_coord.index[i];
            }
            else if (i>=stack_dim)
            {
                src_coord.index[i] = stack_coord.index[i+1];
            }
        }
        // compute src offset
        int64_t src_offset =0;
        for (int64_t i=0; i<input_ndim; i++){
            src_offset += src_coord.index[i] * input_metadatas[tensor_index].stride[i];
        }

        // compute add coord
        add_coord.index[0] = 0;
        add_coord.index[1] = stack_coord.index[1];
        add_coord.index[2] = stack_coord.index[3];
        // compute add offset
        int add_offset = 0;
        for (int i=0; i<3; i++)
        {
            add_offset += add_coord.index[i] * add_metadata.stride[i];
        }

        // compute and copy data from src to dest
        out_metadata.data_ptr[idx] = 
            device_add<in_scalar_t>(input_metadatas[tensor_index].data_ptr[src_offset], add_metadata.data_ptr[add_offset]);
    }
}

template <int ndim, typename scalar_t>
TensorMetadata<ndim, scalar_t> make_tensor_metadata_from_vec(const std::vector<int64_t>& shape, const std::vector<int64_t>& stride, scalar_t* data_ptr)
{
    TensorMetadata<ndim, scalar_t> tensor_metadata;
    tensor_metadata.data_ptr = data_ptr;
    for (int i=0; i<ndim; i++)
    {
        tensor_metadata.stride[i] = stride[i];
        tensor_metadata.size[i] = shape[i];
    }
    return tensor_metadata;
}

template <int ndim, typename scalar_t>
TensorMetadata<ndim, scalar_t> make_tensor_metadata(const torch::Tensor& tensor)
{
    TensorMetadata<ndim, scalar_t> tensor_metadata;
    tensor_metadata.data_ptr = (scalar_t*)tensor.data_ptr();
    for (int i=0; i<ndim; i++)
    {
        tensor_metadata.stride[i] = tensor.stride(i);
        tensor_metadata.size[i] = tensor.size(i);
    }
    return tensor_metadata;
}

template<int input_ndim, int index_ndim, typename in_scalar_t, typename index_scalar_t>
void launch_kernel(torch::Tensor& out, const std::vector<torch::Tensor>& inputs, torch::Tensor& add, torch::Tensor& index, int64_t stack_dim, int64_t index_dim)
{
    int64_t numel = out.numel();
    int64_t num_input = inputs.size();
    std::vector<int64_t> stack_shape = get_stack_out_shape(inputs, stack_dim);
    std::vector<int64_t> stack_stride = get_stride_by_shape(stack_shape);

    // prepare metadata
    auto out_metadata = make_tensor_metadata<input_ndim+index_ndim, in_scalar_t>(out);
    auto index_metadata = make_tensor_metadata<index_ndim, index_scalar_t>(index);
    auto stack_metadata = make_tensor_metadata_from_vec<input_ndim+1, in_scalar_t>(stack_shape, stack_stride, nullptr);
    auto add_metadata = make_tensor_metadata<3, in_scalar_t>(add);
    std::vector<TensorMetadata<input_ndim, in_scalar_t>> input_metadatas;
    for(int i=0; i<num_input; i++)
    {  
        auto input_meta = make_tensor_metadata<input_ndim, in_scalar_t>(inputs[i]);
        input_metadatas.push_back(input_meta);
    }
    TensorMetadata<input_ndim, in_scalar_t>* in_meta_d;
    cudaMalloc(&in_meta_d, num_input * sizeof(TensorMetadata<input_ndim, in_scalar_t>));
    cudaMemcpy(in_meta_d, input_metadatas.data(), num_input * sizeof(TensorMetadata<input_ndim, in_scalar_t>), cudaMemcpyHostToDevice);
    
    // struct used to hold the coord of element
    struct Coord<input_ndim+index_ndim> out_coord;
    struct Coord<input_ndim+1> stack_coord;
    struct Coord<index_ndim> index_coord;
    struct Coord<input_ndim> src_coord;
    struct Coord<3> add_coord;

    int block = 512;
    int grid = (numel + block -1) /block;

    fused_kernel<input_ndim, index_ndim, in_scalar_t, index_scalar_t><<<grid, block>>>(
        out_metadata,
        stack_metadata,
        index_metadata,
        num_input,
        in_meta_d,
        out_coord,
        index_coord,
        stack_coord,
        src_coord,
        add_coord,
        add_metadata,
        stack_dim,
        index_dim,
        numel
        );
    cudaDeviceSynchronize();
    cudaFree(in_meta_d);
}


void fused_stack_add_index_cuda(torch::Tensor& out, const std::vector<torch::Tensor>& inputs, torch::Tensor& add, torch::Tensor& index, int64_t stack_dim, int64_t index_dim){
    int64_t input_ndim = inputs[0].dim();
    int64_t index_ndim = index.dim();
    torch::ScalarType in_scalar_type = inputs[0].scalar_type();
    torch::ScalarType index_scalar_type = index.scalar_type();

#define DISPATCH_BY_INDEX_DTYPE(INPUT_NDIM, INDEX_NDIM, IN_DTYPE) \
    if (index_scalar_type == torch::ScalarType::Long) { \
        launch_kernel<INPUT_NDIM, INDEX_NDIM, IN_DTYPE, int64_t>(out, inputs, add, index, stack_dim, index_dim); \
    } else { \
        launch_kernel<INPUT_NDIM, INDEX_NDIM, IN_DTYPE, int>(out, inputs, add, index, stack_dim, index_dim); \
    }

#define DISPATCH_BY_INPUT_DTYPE(INPUT_NDIM, INDEX_NDIM) \
    if (in_scalar_type == torch::ScalarType::Float) { \
        DISPATCH_BY_INDEX_DTYPE(INPUT_NDIM, INDEX_NDIM, float) \
    } else { \
        DISPATCH_BY_INDEX_DTYPE(INPUT_NDIM, INDEX_NDIM, half) \
    }

#define DISPATCH_BY_INPUT_NDIM(INPUT_NDIM) \
    { \
        if (index_ndim == 1 ) { \
            DISPATCH_BY_INPUT_DTYPE(INPUT_NDIM, 1); \
        } else if (index_ndim == 2) { \
            DISPATCH_BY_INPUT_DTYPE(INPUT_NDIM, 2); \
        } else { \
            DISPATCH_BY_INPUT_DTYPE(INPUT_NDIM, 3); \
        } \
        break; \
    }
  
    switch (input_ndim)
    {   
        case 1: 
            DISPATCH_BY_INPUT_NDIM(1)
        case 2: 
            DISPATCH_BY_INPUT_NDIM(2)
        case 3: 
            DISPATCH_BY_INPUT_NDIM(3)
        default:
            DISPATCH_BY_INPUT_NDIM(4)
    }

}