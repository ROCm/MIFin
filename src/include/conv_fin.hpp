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
#ifndef GUARD_MIOPEN_CONV_FIN_HPP
#define GUARD_MIOPEN_CONV_FIN_HPP

#include "input_flags.hpp"
#include "fin.hpp"
#include "tensor.hpp"
#include "random.hpp"

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

namespace fin {

template <typename Tgpu, typename Tcpu>
class ConvFin : public Fin
{
    public:
    ConvFin() : Fin()
    {
#if 0
        // constructor of the tensor class takes care of this
        miopenCreateTensorDescriptor(&inputTensor);
        miopenCreateTensorDescriptor(&weightTensor);
        miopenCreateTensorDescriptor(&outputTensor);
        miopenCreateTensorDescriptor(&biasTensor);
        miopenCreateTensorDescriptor(&inputTensor_vect4);
        miopenCreateTensorDescriptor(&weightTensor_vect4);
        miopenCreateConvolutionDescriptor(&convDesc);
// ignore the warmup stuff, we deal with it in the main loop
        {
            AutoMiopenWarmupMode warmupMode;
            miopenCreateTensorDescriptor(&warmupInputTensor);
            miopenCreateTensorDescriptor(&warmupWeightTensor);
            miopenCreateTensorDescriptor(&warmupOutputTensor);
            miopenCreateConvolutionDescriptor(&warmupConvDesc);
        }
#endif
        // workspace_dev = nullptr; // TODO: replaced with a tensor class
        // the variable name is implementation dependent, checking size instead
        InitDataType<Tgpu>();
    }
    int AddCmdLineArgs();
    int ParseCmdLineArgs(int argc, char* argv[]);
    InputFlags& GetInputFlags() { return inflags; }

    std::vector<int> GetInputTensorLengthsFromCmdLine();
    std::vector<int> GetWeightTensorLengthsFromCmdLine();
    std::vector<int> GetBiasTensorLengthsFromCmdLine();
    int SetConvDescriptorFromCmdLineArgs();
    std::vector<size_t> GetOutputTensorLengths() const ;
    int AllocateBuffersAndCopy();
    int FindForward(int& ret_algo_count,
            std::vector<miopenConvAlgoPerf_t>& perf_results);
    int RunForwardGPU();
    int RunBackwardGPU() { return 0;}
    int GetandSetData();
    bool IsInputTensorTransform() const;
    ~ConvFin() {}
    InputFlags inflags;

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

    miopenDataType_t data_type = miopenFloat;
};

template <typename Tgpu, typename Tref>
int ConvFin<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    // TODO: add an argument for the json file or figure how to read it from a pipe (|)
    inflags.Parse(argc, argv);

    auto num_iterations = inflags.GetValueInt("iter");
#if 0
    if(num_iterations < 1)
    {
        std::cout << "Fatal: Number of iterations must be > 0: " << num_iterations << std::endl;
        return 1;
    }
#endif
    auto time_enabled = (inflags.GetValueInt("time") != 0);
    int wall_enabled = 0;
    int warmup_enabled = 0;
    {
        const int val = inflags.GetValueInt("wall");
        if(val >= 1)
        {
            if(!time_enabled)
            {
                std::cout << "Info: '--wall " << val << "' is ignored because '--time' is not set"
                          << std::endl;
            }
            else
            {
                wall_enabled   = (val >= 1);
                warmup_enabled = (val >= 2);
            }
        }
    }

    if(time_enabled)
    {
        handle.EnableProfiling(true);
    }

    is_fwd = (inflags.GetValueInt("forw") == 0 || inflags.GetValueInt("forw") & 1);
    is_bwd = (inflags.GetValueInt("forw") == 0 || inflags.GetValueInt("forw") & 2);
    is_wrw = (inflags.GetValueInt("forw") == 0 || inflags.GetValueInt("forw") & 4);

    const auto solution_value = inflags.GetValueInt("solution");

    if(solution_value >= 0)
        immediate_solution = solution_value;

    return 0;
}

template <typename Tgpu, typename Tref>
int ConvFin<Tgpu, Tref>::GetandSetData()
{
    return (0);
}

template <typename Tgpu, typename Tref>
int ConvFin<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag(
        "spatial_dim", '_', "2", "convolution spatial dimension (Default-2)", "int");
    inflags.AddInputFlag("forw",
                         'F',
                         "0",
                         "Flag enables fwd, bwd, wrw convolutions"
                         "\n0 fwd+bwd+wrw (default)"
                         "\n1 fwd only"
                         "\n2 bwd only"
                         "\n4 wrw only"
                         "\n3 fwd+bwd"
                         "\n5 fwd+wrw"
                         "\n6 bwd+wrw",
                         "int");
    inflags.AddInputFlag("batchsize", 'n', "100", "Mini-batch size (Default=100)", "int");
    inflags.AddInputFlag("in_channels", 'c', "3", "Number of Input Channels (Default=3)", "int");
    inflags.AddInputFlag("in_d", '!', "32", "Input Depth (Default=32)", "int");
    inflags.AddInputFlag("in_h", 'H', "32", "Input Height (Default=32)", "int");
    inflags.AddInputFlag("in_w", 'W', "32", "Input Width (Default=32)", "int");
    inflags.AddInputFlag(
        "out_channels", 'k', "32", "Number of Output Channels (Default=32)", "int");
    inflags.AddInputFlag("fil_d", '@', "3", "Filter Depth (Default=3)", "int");
    inflags.AddInputFlag("fil_h", 'y', "3", "Filter Height (Default=3)", "int");
    inflags.AddInputFlag("fil_w", 'x', "3", "Filter Width (Default=3)", "int");
    inflags.AddInputFlag(
        "conv_stride_d", '#', "1", "Convolution Stride for Depth (Default=1)", "int");
    inflags.AddInputFlag(
        "conv_stride_h", 'u', "1", "Convolution Stride for Height (Default=1)", "int");
    inflags.AddInputFlag(
        "conv_stride_w", 'v', "1", "Convolution Stride for Width (Default=1)", "int");
    inflags.AddInputFlag("pad_d", '$', "0", "Zero Padding for Depth (Default=0)", "int");
    inflags.AddInputFlag("pad_h", 'p', "0", "Zero Padding for Height (Default=0)", "int");
    inflags.AddInputFlag("pad_w", 'q', "0", "Zero Padding for Width (Default=0)", "int");
    inflags.AddInputFlag("pad_val", 'r', "0", "Padding Value (Default=0)", "int");
    inflags.AddInputFlag(
        "trans_output_pad_d", '%', "0", "Zero Padding Output for Depth (Default=0)", "int");
    inflags.AddInputFlag(
        "trans_output_pad_h", 'Y', "0", "Zero Padding Output for Height (Default=0)", "int");
    inflags.AddInputFlag(
        "trans_output_pad_w", 'X', "0", "Zero Padding Output for Width (Default=0)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("verification_cache",
                         'C',
                         "",
                         "Use specified directory to cache verification data. Off by default.",
                         "string");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag("wall",
                         'w',
                         "0",
                         "Wall-clock Time Each Layer"
                         "\n0 Off (Default)"
                         "\n1 On, requires '--time 1')"
                         "\n2 On, warm-up the library (prefetch db caches), requires '--time 1')",
                         "int");
    inflags.AddInputFlag("search", 's', "0", "Search Kernel Config (Default=0)", "int");
    inflags.AddInputFlag("printconv", 'P', "1", "Print Convolution Dimensions (Default=1)", "int");
    inflags.AddInputFlag("dump_output", 'o', "0", "Dumps the output buffers (Default=0)", "int");
    inflags.AddInputFlag("in_data", 'd', "", "Input data filename (Default=)", "string");
    inflags.AddInputFlag("weights", 'e', "", "Input weights filename (Default=)", "string");
    inflags.AddInputFlag("bias", 'b', "", "Use Bias (Default=0)", "int");
    inflags.AddInputFlag(
        "mode", 'm', "conv", "Convolution Mode (conv, trans) (Default=conv)", "str");

    inflags.AddInputFlag(
        "pad_mode", 'z', "default", "Padding Mode (same, valid, default) (Default=default)", "str");
    inflags.AddInputFlag("tensor_vect",
                         'Z',
                         "0",
                         "tensor vectorization type (none, vect_c, vect_n) (Default=0)",
                         "int");
    inflags.AddInputFlag("dilation_d", '^', "1", "Dilation of Filter Depth (Default=1)", "int");
    inflags.AddInputFlag("dilation_h", 'l', "1", "Dilation of Filter Height (Default=1)", "int");
    inflags.AddInputFlag("dilation_w", 'j', "1", "Dilation of Filter Width (Default=1)", "int");
    inflags.AddInputFlag("in_bias", 'a', "", "Input bias filename (Default=)", "string");
    inflags.AddInputFlag("group_count", 'g', "1", "Number of Groups (Default=1)", "int");
    inflags.AddInputFlag("dout_data",
                         'D',
                         "",
                         "dy data filename for backward weight computation (Default=)",
                         "string");
    inflags.AddInputFlag("solution",
                         'S',
                         "-1",
                         "Use immediate mode, run solution with specified id."
                         "\nAccepts integer argument N:"
                         "\n=0 Immediate mode, build and run fastest solution"
                         "\n>0 Immediate mode, build and run solution_id = N"
                         "\n<0 Use Find() API (Default=-1)",
                         "int");

    return 0;
}

template <typename Tgpu, typename Tref>
std::vector<int> ConvFin<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine()
{
    std::vector<int> in_lens;

    int spatial_dim = inflags.GetValueInt("spatial_dim");
    in_lens.resize(2 + spatial_dim);

    in_lens[0] = inflags.GetValueInt("batchsize");
    in_lens[1] = inflags.GetValueInt("in_channels");

    auto in_spatial_lens = boost::adaptors::slice(in_lens, 2, 2 + spatial_dim);

    if(spatial_dim == 2)
    {
        in_spatial_lens[0] = inflags.GetValueInt("in_h");
        in_spatial_lens[1] = inflags.GetValueInt("in_w");
    }
    else if(spatial_dim == 3)
    {
        in_spatial_lens[0] = inflags.GetValueInt("in_d");
        in_spatial_lens[1] = inflags.GetValueInt("in_h");
        in_spatial_lens[2] = inflags.GetValueInt("in_w");
    }
    else
    {
        MIOPEN_THROW("unsupported convolution dimension");
    }

    return in_lens;
}

template <typename Tgpu, typename Tref>
std::vector<int> ConvFin<Tgpu, Tref>::GetWeightTensorLengthsFromCmdLine()
{
    std::vector<int> wei_lens;

    int spatial_dim = inflags.GetValueInt("spatial_dim");
    wei_lens.resize(2 + spatial_dim);

    auto wei_spatial_lens = boost::adaptors::slice(wei_lens, 2, 2 + spatial_dim);

    int group_count = std::max(inflags.GetValueInt("group_count"), 1);

    int wei_k_len = inflags.GetValueInt("out_channels");
    int wei_c_len = inflags.GetValueInt("in_channels");

    if(spatial_dim == 2)
    {
        wei_spatial_lens[0] = inflags.GetValueInt("fil_h");
        wei_spatial_lens[1] = inflags.GetValueInt("fil_w");
    }
    else if(spatial_dim == 3)
    {
        wei_spatial_lens[0] = inflags.GetValueInt("fil_d");
        wei_spatial_lens[1] = inflags.GetValueInt("fil_h");
        wei_spatial_lens[2] = inflags.GetValueInt("fil_w");
    }
    else
    {
        MIOPEN_THROW("unsupported convolution dimension");
    }

    if(group_count > 1)
    {
        if(wei_c_len % group_count != 0 || wei_k_len % group_count != 0 ||
           group_count > wei_c_len || group_count > wei_k_len)
        {
            MIOPEN_THROW("Invalid group number\n");
        }
    }

    miopenConvolutionMode_t mode;
    if((inflags.GetValueStr("mode")) == "conv")
    {
        mode = miopenConvolution;
    }
    else if((inflags.GetValueStr("mode")) == "trans")
    {
        mode = miopenTranspose;
    }
    else
    {
        MIOPEN_THROW("Incorrect Convolution Mode\n");
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
std::vector<int> ConvFin<Tgpu, Tref>::GetBiasTensorLengthsFromCmdLine()
{
    int spatial_dim = inflags.GetValueInt("spatial_dim");

    std::vector<int> bias_lens(2 + spatial_dim, 1);

    bias_lens[1] = inflags.GetValueInt("out_channels");

    return bias_lens;
}

template <typename Tgpu, typename Tref>
int ConvFin<Tgpu, Tref>::SetConvDescriptorFromCmdLineArgs()
{
    int spatial_dim = inflags.GetValueInt("spatial_dim");

    std::vector<int> in_spatial_lens(spatial_dim);
    std::vector<int> wei_spatial_lens(spatial_dim);
    std::vector<int> pads(spatial_dim);
    std::vector<int> conv_strides(spatial_dim);
    std::vector<int> conv_dilations(spatial_dim);
    std::vector<int> trans_output_pads(spatial_dim);

    if(spatial_dim == 2)
    {
        in_spatial_lens[0]   = inflags.GetValueInt("in_h");
        in_spatial_lens[1]   = inflags.GetValueInt("in_w");
        wei_spatial_lens[0]  = inflags.GetValueInt("fil_h");
        wei_spatial_lens[1]  = inflags.GetValueInt("fil_w");
        pads[0]              = inflags.GetValueInt("pad_h");
        pads[1]              = inflags.GetValueInt("pad_w");
        conv_strides[0]      = inflags.GetValueInt("conv_stride_h");
        conv_strides[1]      = inflags.GetValueInt("conv_stride_w");
        conv_dilations[0]    = inflags.GetValueInt("dilation_h");
        conv_dilations[1]    = inflags.GetValueInt("dilation_w");
        trans_output_pads[0] = inflags.GetValueInt("trans_output_pad_h");
        trans_output_pads[1] = inflags.GetValueInt("trans_output_pad_w");
    }
    else if(spatial_dim == 3)
    {
        in_spatial_lens[0]   = inflags.GetValueInt("in_d");
        in_spatial_lens[1]   = inflags.GetValueInt("in_h");
        in_spatial_lens[2]   = inflags.GetValueInt("in_w");
        wei_spatial_lens[0]  = inflags.GetValueInt("fil_d");
        wei_spatial_lens[1]  = inflags.GetValueInt("fil_h");
        wei_spatial_lens[2]  = inflags.GetValueInt("fil_w");
        pads[0]              = inflags.GetValueInt("pad_d");
        pads[1]              = inflags.GetValueInt("pad_h");
        pads[2]              = inflags.GetValueInt("pad_w");
        conv_strides[0]      = inflags.GetValueInt("conv_stride_d");
        conv_strides[1]      = inflags.GetValueInt("conv_stride_h");
        conv_strides[2]      = inflags.GetValueInt("conv_stride_w");
        conv_dilations[0]    = inflags.GetValueInt("dilation_d");
        conv_dilations[1]    = inflags.GetValueInt("dilation_h");
        conv_dilations[2]    = inflags.GetValueInt("dilation_w");
        trans_output_pads[0] = inflags.GetValueInt("trans_output_pad_d");
        trans_output_pads[1] = inflags.GetValueInt("trans_output_pad_h");
        trans_output_pads[2] = inflags.GetValueInt("trans_output_pad_w");
    }
    else
    {
        MIOPEN_THROW("unsupported convolution dimension");
    }

    int out_c       = inflags.GetValueInt("out_channels");
    int in_c        = inflags.GetValueInt("in_channels");
    int group_count = std::max(inflags.GetValueInt("group_count"), 1);

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
    if((inflags.GetValueStr("mode")) == "conv")
    {
        c_mode = miopenConvolution;
    }
    else if((inflags.GetValueStr("mode")) == "trans")
    {
        c_mode = miopenTranspose;
    }
    else
    {
        printf("Incorrect Convolution Mode\n");
        exit(0);
    }

    miopenPaddingMode_t p_mode;
    if((inflags.GetValueStr("pad_mode")) == "same")
        p_mode = miopenPaddingSame;
    else if((inflags.GetValueStr("pad_mode")) == "valid")
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

// TODO: remove this function in a refactoring pass
template <typename Tgpu, typename Tref>
std::vector<size_t> ConvFin<Tgpu, Tref>::GetOutputTensorLengths() const
{
    return convDesc.GetForwardOutputTensor(inputTensor.desc, weightTensor.desc).GetLengths();
}

template <typename Tgpu, typename Tref>
bool ConvFin<Tgpu, Tref>::IsInputTensorTransform() const
{
    return (data_type == miopenInt8 && inflags.GetValueInt("in_channels") % 4 != 0) ||
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

template <typename Tgpu, typename Tref>
int ConvFin<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    bool is_transform = IsInputTensorTransform();
    bool is_int8      = data_type == miopenInt8 || data_type == miopenInt8x4;
    SetConvDescriptorFromCmdLineArgs();
    std::vector<int> in_len  = GetInputTensorLengthsFromCmdLine();
    std::vector<int> wei_len = GetWeightTensorLengthsFromCmdLine();
    auto out_len = GetOutputTensorLengths();



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
    /* Unless seed is persistent between runs validation using cache stored in file is impossible.
     */
    srand(0);
    auto in_f = [&](auto idx) { 
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
    auto out_f = [&](auto idx) { 
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
    auto wei_f = [&](auto idx) { 
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
    auto bias_f = [&](auto idx) { 
        if(is_int8)
            return static_cast<float>(idx % 8) + RAN_GEN<float>(static_cast<float>(0.0), static_cast<float>(1.0));
        else
            return static_cast<Tgpu>(idx % 8) + RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        };
    inputTensor = tensor<Tgpu, Tref>{q, in_len, (is_fwd || is_wrw), is_bwd, in_f};
    weightTensor = tensor<Tgpu, Tref>{q, wei_len, (is_fwd || is_bwd), is_wrw, wei_f};
    outputTensor = tensor<Tgpu, Tref>{q, out_len, ( is_bwd || is_wrw), is_fwd, out_f};
    if(inflags.GetValueInt("bias") != 0)
    {
        std::vector<int> bias_len = GetBiasTensorLengthsFromCmdLine();
        biasTensor = tensor<Tgpu, Tref>{q, bias_len, true, true, bias_f};
        dbiasTensor = tensor<Tgpu, Tref>{q, bias_len, true, true, bias_f};
    }
    size_t ws_sizeof_find_fwd = 0;
    size_t ws_sizeof_find_wrw = 0;
    size_t ws_sizeof_find_bwd = 0;
    // TODO: Revisit this
    // I am not convinced that we need to allocate ws similar to find mode
    // since the main loop for fin is more like immediate mode on steroids
    if(!immediate_solution)
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
            workspace = tensor<Tgpu, Tref>{q, std::vector<unsigned int>{static_cast<int>(std::ceil(wsSizeof / sizeof(Tgpu)))}, true, true, [](auto idx){ return static_cast<Tgpu>(0);}};
    }
#if 0
    if(inflags.GetValueInt("tensor_vect") == 1 && data_type == miopenInt8)
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
int ConvFin<Tgpu, Tref>::FindForward(int& ret_algo_count,
                                        std::vector<miopenConvAlgoPerf_t>& perf_results)
{
    bool is_transform = IsInputTensorTransform();
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
                            (inflags.GetValueInt("search") == 1) ? true : false);
                            record,
                            ctx,
                            use_winograd_only);
                    });
        }

        if(perf_db.empty())
            MIOPEN_THROW("Fwd Convolution cannot be executed due to incorrect params");

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
                (inflags.GetValueInt("search") == 1) ? true : false);
#endif
    }
    return 0;
}


} // namespace fin
#endif // GUARD_MIOPEN_CONV_FIN_HPP
