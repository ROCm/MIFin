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

#include "conv_fin.hpp"

#include <boost/range/adaptors.hpp>

#define TUNER_THROW(x) throw std::runtime_error(x)

namespace fin {
#if 0
ConvFin::ConvFin() : Fin()
{
}

int ConvFin::AddCmdLineArgs()
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
    inflags.AddInputFlag("iter", 'i', "10", "Number of Find Iterations (Default=10)", "int");
    inflags.AddInputFlag("search", 's', "0", "Search Kernel Config (Default=0)", "int");
    inflags.AddInputFlag("printconv", 'P', "1", "Print Convolution Dimensions (Default=1)", "int");
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
    return 0;
}

std::vector<int> ConvFin::GetInputTensorLengthsFromCmdLine()
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
        TUNER_THROW("unsupported convolution dimension");
    }

    return in_lens;
}

std::vector<int> ConvFin::GetWeightTensorLengthsFromCmdLine()
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
        TUNER_THROW("unsupported convolution dimension");
    }

    if(group_count > 1)
    {
        if(wei_c_len % group_count != 0 || wei_k_len % group_count != 0 ||
           group_count > wei_c_len || group_count > wei_k_len)
        {
            TUNER_THROW("Invalid group number\n");
        }
    }
    // TODO: Just set the conv desc once and use it later instead of comparing the flags again and again
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
        TUNER_THROW("Incorrect Convolution Mode\n");
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

std::vector<int> ConvFin::GetBiasTensorLengthsFromCmdLine()
{
    int spatial_dim = inflags.GetValueInt("spatial_dim");

    std::vector<int> bias_lens(2 + spatial_dim, 1);

    bias_lens[1] = inflags.GetValueInt("out_channels");

    return bias_lens;
}

int ConvFin::SetConvDescriptorFromCmdLineArgs()
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
        TUNER_THROW("unsupported convolution dimension");

    int out_c       = inflags.GetValueInt("out_channels");
    int in_c        = inflags.GetValueInt("in_channels");
    int group_count = std::max(inflags.GetValueInt("group_count"), 1);

    if(group_count > 1)
    {
        if(in_c % group_count != 0 || out_c % group_count != 0 || group_count > in_c ||
           group_count > out_c)
        {
            std::cout << "Invalid group number\n";
            exit(0);
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
        TUNER_THROW("Incorrect Convolution Mode");

    // adjust padding based on user-defined padding mode
    if(mode == miopenConvolution &&
       (miopen::all_of(conv_dilations, [](auto v) { return v == 1; }) ||
        miopen::all_of(wei_spatial_lens, [](auto v) { return v == 1; })))
    {
        if((inflags.GetValueStr("pad_mode")) == "same")
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
        else if((inflags.GetValueStr("pad_mode")) == "valid")
        {
            for(int i = 0; i < spatial_dim; ++i)
            {
                pads[i] = 0;
            }
        }
    }

    convDesc = miopen::ConvolutionDescriptor(
        spatial_dim,
        mode,
        miopenPaddingDefault,
        std::vector<int>(pads.data(), pads.data() + spatial_dim),
        std::vector<int>(conv_strides.data(), conv_strides.data() + spatial_dim),
        std::vector<int>(conv_dilations.data(), conv_dilations.data() + spatial_dim),
        std::vector<int>(spatial_dim, 0),
        1,
        1.0);

    convDesc.group_count = group_count;

    if(mode == miopenTranspose)
    {
        if(spatial_dim != convDesc.GetSpatialDimension())
        {
            TUNER_THROW("spatialDim not consistent with convolution descriptor");
        }

        std::copy_n(trans_output_pads.data(), spatial_dim, convDesc.trans_output_pads.begin());
    }

    return miopenStatusSuccess;
}

std::vector<int> ConvFin::GetOutputTensorLengths()
{
    return convDesc.GetForwardOutputTensor(inputTensor.desc, weightTensor.desc);
}

int ConvFin::AllocateBuffersAndCopy()
{
    // Timers for benchmarking alloc and copy

    bool is_transform           = false; // TODO: figure out the transform IsInputTensorTransform();
    bool is_int8                = data_type == miopenInt8 || data_type == miopenInt8x4;
    size_t in_sz                = inputTensor.size();
    size_t wei_sz               = weightTensor.size();
    size_t out_sz               = outputTensor.size();
    size_t workSpaceSize_fwd    = 0;
    size_t workSpaceSize_bwd_wt = 0;
    size_t workSpaceSize_bwd_dt = 0;

    if(wrw_allowed)
    {
        workSpaceSize_bwd_wt = convDesc.BackwardWeightsGetWorkSpaceSize(
            GetHandle(),
            convDesc.mode == miopenTranspose ? inputTensor.desc : outputTensor.desc,
            convDesc.mode == miopenTranspose ? outputTensor.desc : inputTensor.desc,
            weightTensor.desc);
    }
    if(bwd_allowed)
    {
        workSpaceSize_bwd_dt = convDesc.mode == miopenTranspose
                                   ? convDesc.ForwardGetWorkSpaceSize(
                                         GetHandle(), weightTensor.desc, outputTensor.desc, inputTensor.desc)
                                   : convDesc.BackwardDataGetWorkSpaceSize(
                                         GetHandle(), weightTensor.desc, outputTensor.desc, inputTensor.desc);
    }
    if(forward_allowed)
    {
        miopen::TensorDescriptor& wDesc = is_transform ? weightTensor_vect4 : weightTensor.desc;
        miopen::TensorDescriptor& xDesc = is_transform ? inputTensor_vect4 : inputTensor.desc;

        workSpaceSize_fwd =
            convDesc.mode == miopenTranspose
                ? convDesc.BackwardDataGetWorkSpaceSize(GetHandle(), wDesc, xDesc, outputTensor)
                : convDesc.ForwardGetWorkSpaceSize(GetHandle(), wDesc, xDesc, outputTensor);
    }

// TODO: Is this still valid ? 
#if 0
    // Workaround: Pad buffers allocations to be a multiple of 2M
    if(miopen::IsEnabled(MIOPEN_DRIVER_PAD_BUFFERS_2M{}))
    {
        PadBufferSize(wei_sz, sizeof(Tgpu));
        PadBufferSize(out_sz, sizeof(Tgpu));
    }
#endif

    size_t workSpaceNbVal_bwd_dt = workSpaceSize_bwd_dt / sizeof(Tgpu);
    size_t workSpaceNbVal_bwd_wt = workSpaceSize_bwd_wt / sizeof(Tgpu);
    size_t workSpaceNbVal_fwd    = workSpaceSize_fwd / sizeof(Tgpu);
    // TODO: tensor ctor takes care of this
#if 0
    in_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    din_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    wei_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, wei_sz, sizeof(Tgpu)));
    dwei_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, wei_sz, sizeof(Tgpu)));
    dout_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
    out_dev =
        std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, is_int8 ? sizeof(float) : sizeof(Tgpu)));
#endif 
    if(workSpaceSize_bwd_dt != 0)
    {
        workspace_bwd_data_dev =
            std::unique_ptr<GPUMem>(new GPUMem(ctx, workSpaceNbVal_bwd_dt, sizeof(Tgpu)));
        workspace_bwd_data      = std::vector<Tgpu>(workSpaceNbVal_bwd_dt, static_cast<Tgpu>(0));
        workspace_bwd_data_host = std::vector<Tref>(workSpaceNbVal_bwd_dt, static_cast<Tref>(0));
    }
    if(workSpaceSize_bwd_wt != 0)
    {
        workspace_bwd_weights_dev =
            std::unique_ptr<GPUMem>(new GPUMem(ctx, workSpaceNbVal_bwd_wt, sizeof(Tgpu)));
        workspace_bwd_weights      = std::vector<Tgpu>(workSpaceNbVal_bwd_wt, static_cast<Tgpu>(0));
        workspace_bwd_weights_host = std::vector<Tref>(workSpaceNbVal_bwd_wt, static_cast<Tref>(0));
    }
    if(workSpaceSize_fwd != 0)
    {
        workspace_fwd_dev =
            std::unique_ptr<GPUMem>(new GPUMem(ctx, workSpaceNbVal_fwd, sizeof(Tgpu)));
        workspace_fwd      = std::vector<Tgpu>(workSpaceNbVal_fwd, static_cast<Tgpu>(0));
        workspace_fwd_host = std::vector<Tref>(workSpaceNbVal_fwd, static_cast<Tref>(0));
    }

    inputTensor = tensor{ctx, cpuDataType, gpuDataType, i

    in   = tensor<Tgpu>(inputTensor.GetLengths());
    din  = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    din_host  = tensor<Tref>(inputTensor.desc.GetLengths());

    wei  = tensor<Tgpu>(weightTensor.GetLengths());
    dwei = std::vector<Tgpu>(wei_sz, static_cast<Tgpu>(0));
    dwei_host = tensor<Tref>(weightTensor.desc.GetLengths());

    out  = tensor<Tgpu>(outputTensor.GetLengths());
    dout = tensor<Tgpu>(outputTensor.GetLengths());
    outhost   = tensor<Tref>(outputTensor.desc.GetLengths());

    if(is_int8)
        out_int8 = std::vector<float>(out_sz, static_cast<float>(0));
    if(is_transform)
    {
        in_vect4_dev = std::unique_ptr<GPUMem>(
            new GPUMem(ctx, GetTensorSize(inputTensor_vect4), sizeof(Tgpu)));
        wei_vect4_dev = std::unique_ptr<GPUMem>(
            new GPUMem(ctx, GetTensorSize(weightTensor_vect4), sizeof(Tgpu)));
    }


    /* Unless seed is persistent between runs validation using cache stored in file is impossible.
     */
    srand(0);

    if(is_int8)
    {
        float Data_scale = 127.0;

        for(int i = 0; i < in_sz; i++)
        {
            in.data[i] = static_cast<Tgpu>(Data_scale * static_cast<float>(1.0));
        }

        if(inflags.GetValueInt("bias") != 0)
        {
            size_t b_sz = biasTensor.size();
            b_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, b_sz, sizeof(float)));
            b_int8      = std::vector<float>(b_sz, static_cast<float>(0));
            for(int i = 0; i < b_sz; i++)
            {
                b_int8[i] = static_cast<float>(i % 8) + static_cast<float>(1.0);
            }

            b_dev->ToGPU(q, b_int8.data());
        }

        for(int i = 0; i < wei_sz; i++)
        {
            wei.data[i] = static_cast<Tgpu>(Data_scale * 2 * detail::RanGenWeights<float>());
        }
    }
    else
    {
        Tgpu Data_scale = static_cast<Tgpu>(0.01);

        for(int i = 0; i < in_sz; i++)
        {
            in.data[i] = Data_scale * static_cast<Tgpu>(1.0);
        }

        for(int i = 0; i < out_sz; i++)
        {
            dout.data[i] = Data_scale * static_cast<Tgpu>(1.0);
        }

        if(inflags.GetValueInt("bias") != 0)
        {
            size_t b_sz = GetTensorSize(biasTensor);
            b_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, b_sz, sizeof(Tgpu)));
            db_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, b_sz, sizeof(Tgpu)));
            b           = tensor<Tgpu>(biasTensor.GetLengths());
            db          = std::vector<Tgpu>(b_sz, static_cast<Tgpu>(0));
            db_host     = tensor<Tref>(biasTensor.GetLengths());
            for(int i = 0; i < b_sz; i++)
            {
                b.data[i] = static_cast<Tgpu>(i % 8) + static_cast<Tgpu>(1.0);
                db[i]     = static_cast<Tgpu>(i % 8) + static_cast<Tgpu>(1.0);
            }

            b_dev->ToGPU(q, b.data.data());
            db_dev->ToGPU(q, db.data());
        }

        for(int i = 0; i < wei_sz; i++)
        {
            wei.data[i] = Data_scale * detail::RanGenWeights<Tgpu>();
        }
    }

#if FIN_BACKEND_OPENCL
    cl_int status;
#elif FIN_BACKEND_HIP
#define CL_SUCCESS 0
    int status;
#endif
    status = in_dev->ToGPU(q, in.data.data());
    status |= din_dev->ToGPU(q, din.data());
    status |= wei_dev->ToGPU(q, wei.data.data());
    status |= dwei_dev->ToGPU(q, dwei.data());
    status |= dout_dev->ToGPU(q, dout.data.data());
    status |= (is_int8 ? out_dev->ToGPU(q, out_int8.data()) : out_dev->ToGPU(q, out.data.data()));
    if(workSpaceSize_bwd_dt != 0)
        status |= workspace_bwd_data_dev->ToGPU(q, workspace_bwd_data.data());
    if(workSpaceSize_bwd_wt != 0)
        status |= workspace_bwd_weights_dev->ToGPU(q, workspace_bwd_weights.data());
    if(workSpaceSize_fwd != 0)
        status |= workspace_fwd_dev->ToGPU(q, workspace_fwd.data());

    if(status != CL_SUCCESS)
        std::cout << "Error copying data to GPU\n";

    return miopenStatusSuccess;
}


int ConvFin::FindForward(int& ret_algo_count,
                                       int request_algo_count,
                                       std::vector<miopenConvAlgoPerf_t>& perf_results)
{
    bool is_transform = IsInputTensorTransform();

    if(convDesc.mode == miopenTranspose)
    {
        convDesc.FindConvBwdDataAlgorithm(
                GetHandle(),
                (is_transform ? inputTensor_vect4 : inputTensor.desc),
                is_transform ? in_vect4_dev->GetMem() : in_dev->GetMem(),
                (is_transform ? weightTensor_vect4 : weightTensor),
                is_transform ? wei_vect4_dev->GetMem() : wei_dev->GetMem(),
                outputTensor,
                out_dev->GetMem(),
                request_algo_count,
                &ret_algo_count,
                perf_results.data(),
                (workspace_fwd_dev != nullptr) ? workspace_fwd_dev->GetMem() : nullptr,
                (workspace_fwd_dev != nullptr) ? workspace_fwd_dev->GetSize() : 0,
                (inflags.GetValueInt("search") == 1) ? true : false);

        for(int i = 0; i < ret_algo_count; ++i)
        {
            // It is guaranteed that enum values are equal, see conv_algo_name.cpp
            perf_results.data()[i].fwd_algo =
                static_cast<miopenConvFwdAlgorithm_t>(perf_results.data()[i].bwd_data_algo);
        }
    }
    convDesc.FindConvFwdAlgorithm(
            GetHandle(),
            (is_transform ? inputTensor_vect4 : inputTensor.desc),
            is_transform ? in_vect4_dev->GetMem() : in_dev->GetMem(),
            (is_transform ? weightTensor_vect4 : weightTensor),
            is_transform ? wei_vect4_dev->GetMem() : wei_dev->GetMem(),
            outputTensor,
            out_dev->GetMem(),
            request_algo_count,
            &ret_algo_count,
            perf_results.data(),
            (workspace_fwd_dev != nullptr) ? workspace_fwd_dev->GetMem() : nullptr,
            (workspace_fwd_dev != nullptr) ? workspace_fwd_dev->GetSize() : 0,
            (inflags.GetValueInt("search") == 1) ? true : false);
}

int ConvFin::RunForwardGPU()
{
    if(!forward_allowed)
        return 0;

    int ret_algo_count;
    int request_algo_count = 2;
    std::vector<miopenConvAlgoPerf_t> perf_results(request_algo_count);

    bool is_transform = IsInputTensorTransform();
    if(is_transform)
    {
        float aph = 1.0;
        float bta = 0.0;
        TransformTensor(GetHandle(),
                        &aph,
                        inputTensor.desc,
                        in_dev->GetMem(),
                        &bta,
                        inputTensor_vect4,
                        in_vect4_dev->GetMem());

        TransformTensor(GetHandle(),
                        &aph,
                        weightTensor,
                        DataCast(wei_dev->GetMem()),
                        &bta,
                        weightTensor_vect4,
                        DataCast(wei_vect4_dev->GetMem()));
    }

    int iter = inflags.GetValueInt("iter");
    int status;
    for(int i = 0; i < iter; i++)
    {
        status = FindForward(ret_algo_count, request_algo_count, perf_results);
        if(status != miopenStatusSuccess)
            break;
    }
    return miopenStatusSuccess;
}

int ConvFin::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);
    forward_allowed = (inflags.GetValueInt("forw") == 0 || inflags.GetValueInt("forw") & 1);
    bwd_allowed     = (inflags.GetValueInt("forw") == 0 || inflags.GetValueInt("forw") & 2);
    wrw_allowed     = (inflags.GetValueInt("forw") == 0 || inflags.GetValueInt("forw") & 4);
    return 0;
}

int ConvFin::GetAndSetData()
{
#if FIN_BACKEND_OPENCL
    cl_context ctx;

    clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
#elif FIN_BACKEND_HIP
    uint32_t ctx = 0;
#endif
    SetConvDescriptorFromCmdLineArgs();

    std::vector<int> in_len  = GetInputTensorLengthsFromCmdLine();
    std::vector<int> wei_len = GetWeightTensorLengthsFromCmdLine();
    std::vector<int> out_len = GetOutputTensorLengths();

    inputTensor  = tensor{ctx, data_type, in_len};
    weightTensor = tensor{data_type, wei_len};

    if(inflags.GetValueInt("tensor_vect") == 1 && data_type == miopenInt8)
    {
        data_type = miopenInt8x4;
    }

    if(IsInputTensorTransform())
    {
        std::vector<int> in_len_vect4(in_len.begin(), in_len.end()),
            wei_len_vect4(wei_len.begin(), wei_len.end());
        in_len_vect4[1] = ((in_len[1] + 3) / 4) * 4;
        inputTensor_vect4 = tensor{data_type, in_len_vect4};
        wei_len_vect4[1] = ((wei_len[1] + 3) / 4) * 4;
        weightTensor_vect4 = tensor{data_type, wei_len};
    }


    miopenDataType_t y_type =
        (data_type == miopenInt8 || data_type == miopenInt8x4) ? miopenFloat : data_type;
    outputTensor = tensor{y_type, out_len};

    if(inflags.GetValueInt("bias") != 0)
    {
        std::vector<int> bias_len = GetBiasTensorLengthsFromCmdLine();
        biasTensor = tensor{data_type, bias_len};
    }
    return (0);
}
#endif
} // namespace fin
