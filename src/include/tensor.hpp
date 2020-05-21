/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef GUARD_FIN_TENSOR_HPP
#define GUARD_FIN_TENSOR_HPP

#include <half.hpp>
#include <miopen/bfloat16.hpp>


#include <miopen/tensor.hpp>
#include <gpu_mem.hpp>

namespace fin {

using half_float::half;
typedef half float16;

template <typename T>
miopenDataType_t GetDataType();

template <typename T>
miopenDataType_t GetDataType()
{
    static_assert(true,  "Invalid data type");
}

template <typename Tgpu, typename Tcpu>
struct tensor
{
#if FIN_BACKEND_OPENCL
    using context_type = cl_context;
    using accelerator_stream = cl_command_queue;
    cl_command_queue q;

#elif FIN_BACKEND_HIP
    using accelerator_stream = hipStream_t;
    using context_type = uint32_t;
    context_type  ctx = 0;
#endif
    miopen::TensorDescriptor desc;
    std::vector<Tgpu> cpuData; // version of the data for the CPU compute
    std::vector<Tgpu> deviceData; // home for the GPU data on the CPU side
    GPUMem gpuData; // object representing the GPU data ON the GPU
    context_type ctx;
    size_t size()
    {
        return gpuData.GetSize();
    }

    tensor(){}
    template <typename F, typename U>
    tensor(accelerator_stream _q, std::vector<U> _plens, bool is_input, bool is_output, F f)
        : q(_q),  desc(GetDataType<Tgpu>(), _plens) /*,cpuData{size()}, 
            gpuData{_ctx, size(), elem_size()} {}*/
    {
        // Perhaps make is_output and is_input exclusive, however, if -F is 0 then all tensors are both inputs and outputs
#if FIN_BACKEND_OPENCL
        clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
#elif FIN_BACKEND_HIP
        ctx = 0;
#endif
        if(is_input)
            // TODO: check if the datatype is correct;
            cpuData = std::vector<Tgpu>(size(), static_cast<Tgpu>(0));

        if(is_output)
            deviceData = std::vector<Tgpu>(size(), static_cast<Tgpu>(0));

        for(int i = 0; i < size(); i++)
        {
            if(is_input) // is input
                cpuData[i] = f(i);
            // Data_scale * RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
            else /// \ref move_rand
                rand();
        }

        gpuData = GPUMem{ctx, size(), sizeof(Tgpu)};
        int status = 0;
        if(is_input)
            status = gpuData.ToGPU(q, cpuData.data());
        else if(is_output)
            status = gpuData.ToGPU(q, deviceData.data()); // to set the data to zero on the GPU
        // TODO: check status 
        (void)status;
    }
};
} // namespace fin
#if 0
#endif
#endif // GUARD_FIN_TENSOR_HPP
