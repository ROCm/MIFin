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
#ifndef GUARD_CONV_FIN_HPP
#define GUARD_CONV_FIN_HPP

#include "input_flags.hpp"
#include "fin.hpp"
#include "tensor.hpp"
#include "random.hpp"
#include "error.hpp"

#include <miopen/convolution.hpp>
#include <miopen/algorithm.hpp>
#include <miopen/conv/context.hpp>
#include <miopen/conv_solution.hpp>
#include <miopen/solver_id.hpp>
#include <miopen/any_solver.hpp>

#if 0
#include <miopen/tensor_ops.hpp>
#include <miopen/tensor.hpp>
#endif 

#include <boost/range/adaptor/sliced.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <float.h>
#include <fstream>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>
#include <type_traits>
#include <nlohmann/json.hpp>


namespace fin {

using json = nlohmann::json;
// TODO: Create a config class to encapsulate config 
// related code, such as checking direction etc
template <typename Tgpu, typename Tcpu>
class ConvFin : public Fin
{
    public:
    ConvFin(json _job) : Fin()
    {
        job = _job; // TODO: Verify all required fields are present, otherwise throw! 
        command = _job["config"];
        command["bias"] = 0;
        // timing is always enabled
        handle.EnableProfiling(true);
        // TODO: do we need these ?      
        // TODO: What is the default value of direction in the db
        is_fwd = (_job["direction"].get<int>() == 0 || _job["direction"].get<int>() & 1);
        is_bwd = (_job["direction"].get<int>() == 0 || _job["direction"].get<int>() & 2);
        is_wrw = (_job["direction"].get<int>() == 0 || _job["direction"].get<int>() & 4);
        SetConvDescriptor();
        GetandSetData();
        // workspace_dev = nullptr; // TODO: replaced with a tensor class
        // the variable name is implementation dependent, checking size instead
        InitDataType<Tgpu>();
    }

    // Getters and setters
    std::vector<int> GetInputTensorLengths();
    std::vector<int> GetWeightTensorLengths();
    std::vector<int> GetBiasTensorLengths();
    int SetConvDescriptor();
    std::vector<size_t> GetOutputTensorLengths() const ;
    miopenDataType_t GetOutputType() const {return (data_type == miopenInt8 || data_type == miopenInt8x4) ? miopenFloat : data_type;}

    int ProcessStep(const std::string& step_name) override;

    // Steps
    int AllocateBuffers();
    int CalcWorkspace();
    int FillBuffers();
    int CopyToDevice();
    int CopyFromDevice();
    int AllocateBuffersAndCopy();
    int FindForward(int& ret_algo_count,
            std::vector<miopenConvAlgoPerf_t>& perf_results);
    int RunGPU();
    int TestApplicability();
    int GetandSetData();
    bool IsInputTensorTransform() const;
    json command;
    json job;

    tensor<Tgpu, Tcpu> inputTensor;
    tensor<Tgpu, Tcpu> inputTensor_vect4;
    tensor<Tgpu, Tcpu> dinputTensor;
    tensor<Tgpu, Tcpu> outputTensor;
    tensor<Tgpu, Tcpu> doutputTensor;
    tensor<Tgpu, Tcpu> weightTensor;
    tensor<Tgpu, Tcpu> weightTensor_vect4;
    tensor<Tgpu, Tcpu> dweightTensor;
    tensor<Tgpu, Tcpu> biasTensor;
    tensor<Tgpu, Tcpu> dbiasTensor;
    tensor<Tgpu, Tcpu> workspace;
    miopen::ConvolutionDescriptor convDesc;


    bool wrw_allowed = 0, bwd_allowed = 0, forward_allowed = 1;
    bool is_fwd = true;
    bool is_bwd = false;
    bool is_wrw = false; // TODO: check redundancy with above
    int immediate_solution = 0;

};

template<typename Tgpu, typename Tref>
int ConvFin<Tgpu, Tref>::TestApplicability()
{
    // Get a list of the solvers from the solver registry 
    // Create a convolution context and pass to isApplicable and get result
    uint64_t cur_id = 1;
    constexpr uint64_t max_id = 200;
    auto dir = is_fwd ? miopen::conv::Direction::Forward : (is_bwd ? miopen::conv::Direction::BackwardData : miopen::conv::Direction::BackwardWeights);
    miopen::ConvolutionContext ctx{inputTensor.desc, weightTensor.desc, outputTensor.desc, convDesc, dir};
    ctx.SetStream(&handle);
    ctx.DetectRocm();
    std::vector<std::string> app_solvers;
    while(true)
    {
        miopen::solver::Id id(cur_id);
        if(id.IsValid() && id != miopen::solver::Id::gemm() && id != miopen::solver::Id::fft())
        {
            std::cout << cur_id << ": " << id.ToString() << " algo: " << id.GetAlgo(miopen::conv::Direction::Forward) << std::endl;
            auto solver = id.GetSolver();
            try
            {
                if(solver.IsApplicable(ctx))
                {
                    app_solvers.push_back(id.ToString());
                }
            }
            catch(...)
            {
                std::cout << id.ToString() << " raised an exception" << std::endl;
            }
        }
        cur_id++;
        if(cur_id == max_id)
            break;
    }
    output["applicable_solvers"] = app_solvers;
    for(auto& sol : app_solvers)
    {
        std::cout << sol << std::endl;
    }
    return 0;
}

template<typename Tgpu, typename Tref>
int ConvFin<Tgpu, Tref>::RunGPU()
{
    assert(false);
    return 0;
}

template<typename Tgpu, typename Tref>
int ConvFin<Tgpu, Tref>::CopyToDevice()
{
    auto status = inputTensor.ToDevice();
    status |= inputTensor_vect4.ToDevice();
    status |= dinputTensor.ToDevice();
    status |= weightTensor.ToDevice();
    status |= dweightTensor.ToDevice();
    status |= outputTensor.ToDevice();
    status |= doutputTensor.ToDevice();
    status |= biasTensor.ToDevice();
    status |= dbiasTensor.ToDevice();
    status |= workspace.ToDevice();
    return status;
}

template<typename Tgpu, typename Tref>
int ConvFin<Tgpu, Tref>::CopyFromDevice()
{
    auto status = inputTensor.FromDevice();
    status |= inputTensor_vect4.FromDevice();
    status |= dinputTensor.FromDevice();
    status |= weightTensor.FromDevice();
    status |= dweightTensor.FromDevice();
    status |= outputTensor.FromDevice();
    status |= doutputTensor.FromDevice();
    status |= biasTensor.FromDevice();
    status |= dbiasTensor.FromDevice();
    status |= workspace.FromDevice();
    return status;
}

template<typename Tgpu, typename Tref>
int ConvFin<Tgpu, Tref>::ProcessStep(const std::string& step_name)
{
    if(step_name == "alloc_buf")
        return AllocateBuffers();
    if(step_name == "fill_buf")
        return FillBuffers();
    if(step_name == "copy_buf_to_device")
        return CopyToDevice();
    if(step_name == "copy_buf_from_device")
        return CopyFromDevice();
    if(step_name == "applicability")
        return TestApplicability();
    return 0;
}


template <typename Tgpu, typename Tref>
int ConvFin<Tgpu, Tref>::GetandSetData()
{
    auto in_len = GetInputTensorLengths();
    auto wei_len = GetWeightTensorLengths();

    // auto y_type = GetOutputType();

    inputTensor = {handle.GetStream(), in_len, (is_fwd || is_wrw), is_bwd};
    if(is_bwd)
        dinputTensor = {handle.GetStream(), in_len, (is_fwd || is_wrw), true};

    weightTensor = {handle.GetStream(), wei_len, (is_fwd || is_bwd), is_wrw};
    if(is_wrw)
        dweightTensor = {handle.GetStream(), wei_len, (is_fwd || is_bwd), is_wrw};
    // conv, input and weight tensor descriptors need to be set before we can know the 
    // output lengths
    auto out_len = GetOutputTensorLengths();
    outputTensor = {handle.GetStream(), out_len, (is_bwd || is_wrw), is_fwd};
    if(is_bwd || is_wrw)
        doutputTensor = {handle.GetStream(), out_len,(is_bwd || is_wrw) , true};

    if(IsInputTensorTransform())
    {
        std::vector<int> in_len_v4(in_len.begin(), in_len.end());
        in_len_v4[1] = ((in_len[1] + 3) / 4) * 4;
        std::vector<int> wei_len_v4(wei_len.begin(), wei_len.end());
        wei_len_v4[1] = ((wei_len[1] + 3) / 4) * 4;

        inputTensor_vect4 = {handle.GetStream(), in_len_v4, (is_fwd || is_wrw), is_bwd};
        weightTensor_vect4 = {handle.GetStream(), wei_len_v4,(is_fwd || is_bwd), is_wrw};
    }

    // Conv Desc is already setup from the job descriptor


    if(command["bias"].get<int>() != 0)
    {
        auto bias_len = GetBiasTensorLengths();
        biasTensor = {handle.GetStream(), bias_len, true, true};
        dbiasTensor = tensor<Tgpu, Tref>{q, bias_len, true, true};
    }
    // TODO: further investigate the warmpup iteration, I dont think its necessary and can be handled in the main execution loop
    
    return (0);
}

template <typename Tgpu, typename Tref>
std::vector<int> ConvFin<Tgpu, Tref>::GetInputTensorLengths()
{
    std::vector<int> in_lens;

    int spatial_dim = command["spatial_dim"];
    in_lens.resize(2 + spatial_dim);

    in_lens[0] = command["batchsize"];
    in_lens[1] = command["in_channels"];

    auto in_spatial_lens = boost::adaptors::slice(in_lens, 2, 2 + spatial_dim);

    if(spatial_dim == 2)
    {
        in_spatial_lens[0] = command["in_h"];
        in_spatial_lens[1] = command["in_w"];
    }
    else if(spatial_dim == 3)
    {
        in_spatial_lens[0] = command["in_d"];
        in_spatial_lens[1] = command["in_h"];
        in_spatial_lens[2] = command["in_w"];
    }
    else
    {
        FIN_THROW("unsupported convolution dimension");
    }

    return in_lens;
}

template <typename Tgpu, typename Tref>
std::vector<int> ConvFin<Tgpu, Tref>::GetWeightTensorLengths()
{
    std::vector<int> wei_lens;

    int spatial_dim = command["spatial_dim"];
    wei_lens.resize(2 + spatial_dim);

    auto wei_spatial_lens = boost::adaptors::slice(wei_lens, 2, 2 + spatial_dim);

    int group_count = std::max(int(command["group_count"]), 1);

    int wei_k_len = command["out_channels"];
    int wei_c_len = command["in_channels"];

    if(spatial_dim == 2)
    {
        wei_spatial_lens[0] = command["fil_h"];
        wei_spatial_lens[1] = command["fil_w"];
    }
    else if(spatial_dim == 3)
    {
        wei_spatial_lens[0] = command["fil_d"];
        wei_spatial_lens[1] = command["fil_h"];
        wei_spatial_lens[2] = command["fil_w"];
    }
    else
    {
        FIN_THROW("unsupported convolution dimension");
    }

    if(group_count > 1)
    {
        if(wei_c_len % group_count != 0 || wei_k_len % group_count != 0 ||
           group_count > wei_c_len || group_count > wei_k_len)
        {
            FIN_THROW("Invalid group number\n");
        }
    }

    miopenConvolutionMode_t mode;
    if((command["conv_mode"]) == "conv")
    {
        mode = miopenConvolution;
    }
    else if((command["conv_mode"]) == "trans")
    {
        mode = miopenTranspose;
    }
    else
    {
        FIN_THROW("Incorrect Convolution Mode\n");
    }

    if(mode == miopenTranspose)
    {
        wei_lens[0] = wei_c_len;
        wei_lens[1] = wei_k_len / group_count;
    }
    else
    {
        wei_lens[0] = wei_k_len;
        wei_lens[1] = wei_c_len / group_count;
    }

    return wei_lens;
}

template <typename Tgpu, typename Tref>
std::vector<int> ConvFin<Tgpu, Tref>::GetBiasTensorLengths()
{
    int spatial_dim = command["spatial_dim"];

    std::vector<int> bias_lens(2 + spatial_dim, 1);

    bias_lens[1] = command["out_channels"];

    return bias_lens;
}

template <typename Tgpu, typename Tref>
int ConvFin<Tgpu, Tref>::SetConvDescriptor()
{
    size_t spatial_dim = command["spatial_dim"];

    std::vector<int> in_spatial_lens(spatial_dim);
    std::vector<int> wei_spatial_lens(spatial_dim);
    std::vector<int> pads(spatial_dim);
    std::vector<int> conv_strides(spatial_dim);
    std::vector<int> conv_dilations(spatial_dim);
    std::vector<int> trans_output_pads(spatial_dim);

    if(spatial_dim == 2)
    {
        in_spatial_lens[0]   = command["in_h"];
        in_spatial_lens[1]   = command["in_w"];
        wei_spatial_lens[0]  = command["fil_h"];
        wei_spatial_lens[1]  = command["fil_w"];
        pads[0]              = command["pad_h"];
        pads[1]              = command["pad_w"];
        conv_strides[0]      = command["conv_stride_h"];
        conv_strides[1]      = command["conv_stride_w"];
        conv_dilations[0]    = command["dilation_h"];
        conv_dilations[1]    = command["dilation_w"];
        trans_output_pads[0] = 0; // command["trans_output_pad_h"];
        trans_output_pads[1] = 0; // command["trans_output_pad_w"];
    }
    else if(spatial_dim == 3)
    {
        in_spatial_lens[0]   = command["in_d"];
        in_spatial_lens[1]   = command["in_h"];
        in_spatial_lens[2]   = command["in_w"];
        wei_spatial_lens[0]  = command["fil_d"];
        wei_spatial_lens[1]  = command["fil_h"];
        wei_spatial_lens[2]  = command["fil_w"];
        pads[0]              = command["pad_d"];
        pads[1]              = command["pad_h"];
        pads[2]              = command["pad_w"];
        conv_strides[0]      = command["conv_stride_d"];
        conv_strides[1]      = command["conv_stride_h"];
        conv_strides[2]      = command["conv_stride_w"];
        conv_dilations[0]    = command["dilation_d"];
        conv_dilations[1]    = command["dilation_h"];
        conv_dilations[2]    = command["dilation_w"];
        trans_output_pads[0] = 0; // command["trans_output_pad_d"];
        trans_output_pads[1] = 0; // command["trans_output_pad_h"];
        trans_output_pads[2] = 0; // command["trans_output_pad_w"];
    }
    else
    {
        FIN_THROW("unsupported convolution dimension");
    }

    int out_c       = command["out_channels"];
    int in_c        = command["in_channels"];
    int group_count = std::max(int(command["group_count"]), 1);

    if(group_count > 1)
    {
        if(in_c % group_count != 0 || out_c % group_count != 0 || group_count > in_c ||
           group_count > out_c)
        {
            printf("Invalid group number\n");
            exit(0);
        }
    }

    miopenConvolutionMode_t c_mode;
    if((command["conv_mode"]) == "conv")
    {
        c_mode = miopenConvolution;
    }
    else if((command["conv_mode"]) == "trans")
    {
        c_mode = miopenTranspose;
    }
    else
    {
        printf("Incorrect Convolution Mode\n");
        exit(0);
    }

    miopenPaddingMode_t p_mode = miopenPaddingSame;
    if((command["pad_mode"]) == "same")
        p_mode = miopenPaddingSame;
    else if((command["pad_mode"]) == "valid")
        p_mode = miopenPaddingValid;

    // adjust padding based on user-defined padding mode
    if(c_mode == miopenConvolution &&
       (miopen::all_of(conv_dilations, [](auto v) { return v == 1; }) ||
        miopen::all_of(wei_spatial_lens, [](auto v) { return v == 1; })))
    {
        if(p_mode == miopenPaddingSame)
        {
            for(int i = 0; i < spatial_dim; ++i)
            {
                pads[i] =
                    (in_spatial_lens[i] % conv_strides[i] == 0)
                        ? (std::max((wei_spatial_lens[i] - conv_strides[i]), 0))
                        : (std::max((wei_spatial_lens[i] - (in_spatial_lens[i] % conv_strides[i])),
                                    0));
                pads[i] /= 2;
            }
        }
        else if(p_mode == miopenPaddingValid)
        {
            for(int i = 0; i < spatial_dim; ++i)
            {
                pads[i] = 0;
            }
        }
    }

    convDesc = miopen::ConvolutionDescriptor{
        spatial_dim, c_mode, p_mode, pads, conv_strides, conv_dilations, trans_output_pads, group_count};

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<size_t> ConvFin<Tgpu, Tref>::GetOutputTensorLengths() const
{
    return convDesc.GetForwardOutputTensor(inputTensor.desc, weightTensor.desc).GetLengths();
}

template <typename Tgpu, typename Tref>
bool ConvFin<Tgpu, Tref>::IsInputTensorTransform() const
{
    return (data_type == miopenInt8 && int(command["in_channels"]) % 4 != 0) ||
           data_type == miopenInt8x4;
}

namespace detail {

template <typename T>
T RanGenWeights()
{
    return RAN_GEN<T>(static_cast<T>(-0.5), static_cast<T>(0.5));
}

// Shift FP16 distribution towards positive numbers,
// otherwise Winograd FP16 validation fails.
template <>
float16 RanGenWeights()
{
    return RAN_GEN<float16>(static_cast<float16>(-1.0 / 3.0), static_cast<float16>(0.5));
}

} // namespace detail

template<typename Tgpu, typename Tref>
int ConvFin<Tgpu, Tref>::AllocateBuffers()
{
    inputTensor.AllocateBuffers();
    inputTensor_vect4.AllocateBuffers();
    dinputTensor.AllocateBuffers();
    weightTensor.AllocateBuffers();
    dweightTensor.AllocateBuffers();
    outputTensor.AllocateBuffers();
    doutputTensor.AllocateBuffers();
    biasTensor.AllocateBuffers();
    dbiasTensor.AllocateBuffers();
    workspace.AllocateBuffers();
    return 0;
}

template<typename Tgpu, typename Tref>
int ConvFin<Tgpu, Tref>::CalcWorkspace()
{
    // if(solver is known)
    // Find workspace for solver using the GetSolution mechanism
    // else
    //if(!immediate_solution)
    size_t ws_sizeof_find_fwd = 0;
    size_t ws_sizeof_find_wrw = 0;
    size_t ws_sizeof_find_bwd = 0;
    auto is_transform = IsInputTensorTransform();
    {
        if(is_wrw)
            ws_sizeof_find_wrw = convDesc.BackwardWeightsGetWorkSpaceSize(handle, outputTensor.desc, inputTensor.desc, weightTensor.desc);
        if(is_bwd )
        {
            ws_sizeof_find_bwd =  (convDesc.mode == miopenTranspose)
                ? convDesc.ForwardGetWorkSpaceSize(handle, weightTensor.desc, outputTensor.desc, inputTensor.desc)
                : convDesc.BackwardDataGetWorkSpaceSize(handle, weightTensor.desc, outputTensor.desc, inputTensor.desc); 
        }
        if(is_fwd)
        {
            ws_sizeof_find_fwd = (convDesc.mode == miopenTranspose)
                ? convDesc.BackwardDataGetWorkSpaceSize(handle, 
                        (is_transform ? weightTensor_vect4.desc : weightTensor.desc),
                        (is_transform ? inputTensor_vect4.desc : inputTensor.desc),
                        outputTensor.desc)
                : convDesc.ForwardGetWorkSpaceSize(handle,
                        (is_transform ? weightTensor_vect4.desc : weightTensor.desc),
                        (is_transform ? inputTensor_vect4.desc : inputTensor.desc),
                        outputTensor.desc);
        }

        const auto wsSizeof =
            std::max(std::max(ws_sizeof_find_bwd, ws_sizeof_find_wrw), ws_sizeof_find_fwd);
        if(wsSizeof != 0)
            workspace = tensor<Tgpu, Tref>{q, 
                std::vector<unsigned int>{static_cast<unsigned int>(std::ceil(wsSizeof / sizeof(Tgpu)))}, 
                true, false};
        return wsSizeof;
    }
    return  -1;
}

template<typename Tgpu, typename Tref>
int ConvFin<Tgpu, Tref>::FillBuffers()
{
    // TODO: Do we need to initialize d* tensors ? 
    /* Unless seed is persistent between runs validation using cache stored in file is impossible.
     */
    auto is_int8 = (data_type == miopenInt8 || data_type == miopenInt8x4);
    srand(0);
    auto in_f = [&](auto idx) { 
        (void)idx;
        if(is_int8)
        {
            float Data_scale = 127.0;
            return static_cast<Tgpu>(Data_scale * RAN_GEN<float>(static_cast<float>(0.0), 
                                                                 static_cast<float>(1.0)));
        }
        else
        {
            Tgpu Data_scale = static_cast<Tgpu>(0.01);
            return Data_scale * RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }
        };

    inputTensor.FillBuffer(in_f);
    auto out_f = [&](auto idx) { 
        (void)idx;
        if(is_int8)
        {
            return static_cast<Tgpu>(0); // int8 is inference only
        }
        else
        {
            Tgpu Data_scale = static_cast<Tgpu>(0.01);
            return Data_scale * RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }
        };
    outputTensor.FillBuffer(out_f);
    auto wei_f = [&](auto idx) { 
        (void)idx;
        if(is_int8)
        {
            float Data_scale = 127.0;
            return static_cast<Tgpu>(Data_scale * 2 * detail::RanGenWeights<float>());
        }
        else
        {
            Tgpu Data_scale = static_cast<Tgpu>(0.01);
            return Data_scale * detail::RanGenWeights<Tgpu>();
        }
        };
    weightTensor.FillBuffer(wei_f);
    auto bias_f = [&](auto idx) { 
        (void)idx;
        if(is_int8)
            return static_cast<float>(idx % 8) + RAN_GEN<float>(static_cast<float>(0.0), static_cast<float>(1.0));
        else
            return static_cast<Tgpu>(idx % 8) + RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        };
    if(command["bias"].get<int>() != 0)
    {
        biasTensor.FillBuffer(bias_f);
        dbiasTensor.FillBuffer(bias_f);
    }
    return 0;
}

template <typename Tgpu, typename Tref>
int ConvFin<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    // AllocateBuffers();
    // FillBuffers();
    // CopyBuffers(); To GPU
// TODO: Check if this is stil applicable
#if 0
    // Workaround: Pad buffers allocations to be a multiple of 2M
    if(miopen::IsEnabled(MIOPEN_DRIVER_PAD_BUFFERS_2M{}))
    {
        // TODO: remove this, not relevant anymoore
        size_t in_sz      = GetTensorSize(inputTensor);
        size_t wei_sz     = GetTensorSize(weightTensor);
        size_t out_sz     = GetTensorSize(outputTensor);
        // PadBufferSize(in_sz, sizeof(Tgpu));
        PadBufferSize(wei_sz, sizeof(Tgpu));
        PadBufferSize(out_sz, sizeof(Tgpu));
    }
#endif
    // TODO: Revisit this
    // I am not convinced that we need to allocate ws similar to find mode
    // since the main loop for fin is more like immediate mode on steroids
#if 0
    if(command["tensor_vect"] == 1 && data_type == miopenInt8)
    {
        data_type = miopenInt8x4;
    }

    if(IsInputTensorTransform())
    {
        std::vector<int> in_len_vect4(in_len.begin(), in_len.end()),
            wei_len_vect4(wei_len.begin(), wei_len.end());
        in_len_vect4[1] = ((in_len[1] + 3) / 4) * 4;
        SetTensorNd(inputTensor_vect4, in_len_vect4, data_type);
        wei_len_vect4[1] = ((wei_len[1] + 3) / 4) * 4;
        SetTensorNd(weightTensor_vect4, wei_len_vect4, data_type);
    }
#endif
    // TODO: incorporate it
#if 0
    miopenDataType_t y_type =
        (data_type == miopenInt8 || data_type == miopenInt8x4) ? miopenFloat : data_type;
    outputTensor.desc = miopen::TensorDescriptor{y_type, out_len};
#endif
// TODO: fix the fol by specializing the tensor template for int8 and replacing all Tgpu by float
// The packing may be an arg to the tensor ctor
#if 0
    if(is_int8)
        out_int8 = std::vector<float>(out_sz, static_cast<float>(0));
    if(is_transform)
    {
        in_vect4_dev = std::unique_ptr<GPUMem>(
            new GPUMem(ctx, GetTensorSize(inputTensor_vect4), sizeof(Tgpu)));
        wei_vect4_dev = std::unique_ptr<GPUMem>(
            new GPUMem(ctx, GetTensorSize(weightTensor_vect4), sizeof(Tgpu)));
    }
#endif

#if 0
// TODO: only for ref, remove once functionality verified
    if(is_fwd || is_wrw) // is input
        in = tensor<Tgpu>(miopen::deref(inputTensor).GetLengths());
    if(is_bwd) // is output
        din = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    din_host  = tensor<Tref>(miopen::deref(inputTensor).GetLengths());
        for(int i = 0; i < in_sz; i++)
        {
            if(is_fwd || is_wrw) // is input
                in.data[i] =
                    Data_scale * RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
            else /// \ref move_rand
                rand();
        }
    if(is_fwd || is_wrw) // is input
    {
        in_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
        status |= in_dev->ToGPU(q, in.data.data());
    }
    if(is_bwd)
    {
        din_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
        status |= din_dev->ToGPU(q, din.data());
    }
#endif

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ConvFin<Tgpu, Tref>::FindForward(int& /*ret_algo_count*/,
                                        std::vector<miopenConvAlgoPerf_t>& /*perf_results*/)
{
    // bool is_transform = IsInputTensorTransform();
    if(convDesc.mode == miopenTranspose)
    {
    }
    else
    {
        // JD: I pulled this bit out of MIOpen since the default method in MIOpen is Hybrid find which as seen below returns
        // the best solution in the find db, Fin might prefer things a bit differently.
        auto xDesc = inputTensor.desc;
        auto wDesc = weightTensor.desc;
        auto yDesc = outputTensor.desc;
        auto x = inputTensor.gpuData.GetMem();
        auto w = weightTensor.gpuData.GetMem();
        auto y = outputTensor.gpuData.GetMem();
        auto workSpace = workspace.gpuData.GetMem();
        auto workSpaceSize = workspace.gpuData.size();

        const miopen::ProblemDescription problem(xDesc, wDesc, yDesc, convDesc, miopen::conv::Direction::Forward);
        
        auto ctx = miopen::ConvolutionContext{problem};
        ctx.SetStream(&handle);
        ctx.DetectRocm();
        miopen::ConvolutionUserBuffers bufs(workSpace, workSpaceSize);
        bufs.SetFwd(x, w, y);
        ctx.SetBufs(bufs);
        
        // all winograd solvers
        const auto winogradSolvers = convDesc.FindWinogradSolutions(ctx);
        for(const auto& sol : winogradSolvers)
        {
            auto id = miopen::solver::Id{sol.solver_id};
            auto solver = id.GetSolver();
            auto algo_name = id.GetAlgo(miopen::conv::Direction::Forward);
            if(solver.IsApplicable(ctx))
                std::cout << "bingo!" << std::endl;
        }
#if 0
        if((fm.IsFast() || fm.IsHybrid()) && !use_winograd_only)
        {
            size_t count;
            GetForwardSolutions(handle, wDesc, xDesc, yDesc, 1, &count, &sol);
            use_immediate_solution = (count > 0) && !(fm.IsHybrid() && sol.time < 0);
            // In Hybrid Find mode, we use Normal Find instead of Immediate fallback kernels.
        }

        if(use_immediate_solution)
        {
            CompileForwardSolution(handle, wDesc, xDesc, yDesc, sol.solution_id);
            /// It is possible to measure actual execution time and return it to the caller.
            /// \todo Consider if we need (and want to spend time) for this.
            const auto id = solver::Id(sol.solution_id);
            perf_db.push_back(
                    {id.GetAlgo(conv::Direction::Forward), id.ToString(), sol.time, sol.workspace_size});
        }
        else
        {
            perf_db = miopen::UserFindDbRecord::TryLoad(handle, problem, [&](DbRecord& record) {
                    DirConvFindCore(handle, // static function -> compile error
                            inputTensor.desc,
                            x,
                            wDesc,
                            w,
                            yDesc,
                            y,
                            workspace.gpuData.GetMem(),
                            workspace.gpuData.size(),
                            convDesc,
                            (command["search"] == 1) ? true : false);
                            record,
                            ctx,
                            use_winograd_only);
                    });
        }

        if(perf_db.empty())
            FIN_THROW("Fwd Convolution cannot be executed due to incorrect params");

        std::sort(begin(perf_db), end(perf_db));

        for(const auto& entry : perf_db)
            MIOPEN_LOG_I(entry.name << "\t" << entry.time << "\t" << entry.workspace);

        *returnedAlgoCount = std::min(requestAlgoCount, static_cast<int>(perf_db.size()));

        for(int i = 0; i < *returnedAlgoCount; i++)
        {
            perfResults[i].fwd_algo = StringToConvolutionFwdAlgo(perf_db[i].name);
            perfResults[i].time     = perf_db[i].time;
            perfResults[i].memory   = perf_db[i].workspace;
        }

        MIOPEN_LOG_I("FW Chosen Algorithm: " << perf_db[0].solver_id << " , " << perf_db[0].workspace
                << ", "
                << perf_db[0].time);
#endif
#if 0
        const auto rc = miopenFindConvolutionForwardAlgorithm(
                GetHandle(),
                (is_transform ? inputTensor_vect4 : inputTensor),
                (is_transform ? in_vect4_dev->GetMem() : in_dev->GetMem()),
                (is_transform ? weightTensor_vect4 : weightTensor),
                (is_transform ? wei_vect4_dev->GetMem() : wei_dev->GetMem()),
                convDesc,
                outputTensor,
                out_dev->GetMem(),
                request_algo_count,
                &ret_algo_count,
                perf_results.data(),
                workspace_dev != nullptr ? workspace_dev->GetMem() : nullptr,
                ws_sizeof_find_fwd,
                (command["search"] == 1) ? true : false);
#endif
    }
    return 0;
}


} // namespace fin
#endif // GUARD_MIOPEN_CONV_FIN_HPP
