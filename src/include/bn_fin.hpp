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
 * The above copyright notice and this permission notice shall be included in
 *all
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
#ifndef GUARD_BN_FIN_HPP
#define GUARD_BN_FIN_HPP

#include "error.hpp"
#include "fin.hpp"
#include "tensor.hpp"

#include <miopen/execution_context.hpp>
#include <miopen/miopen.h>

#include <miopen/batchnorm/problem_description.hpp>
#include <miopen/batch_norm.hpp>
#include <miopen/batchnorm/invoke_params.hpp>
#include <miopen/batchnorm/solvers.hpp>

#include <nlohmann/json.hpp>

#define EPSILON 1e-3

namespace fin {

using json = nlohmann::json;
// TODO: Create a config class to encapsulate config
// related code, such as checking direction etc
template <typename Tgpu, typename Tcpu>
class BNFin : public Fin
{
    public:
    BNFin() : Fin() {}
    BNFin(json _job) : Fin(), job(_job)
    {
        if(job.contains("config"))
            PrepBatchNorm();
    }

    void VerifyDevProps()
    {
        std::cerr << "Verifying device properties" << std::endl;
        std::string arch    = job["arch"];
        arch                = arch.substr(0, arch.find(':'));
        const size_t num_cu = job["num_cu"];
        std::ignore         = num_cu;
        if(arch == "gfx900")
        {
            assert(num_cu == 56 || num_cu == 64);
        }
        else if(arch == "gfx906")
        {
            assert(num_cu == 60 || num_cu == 64);
        }
        else if(arch == "gfx908")
        {
            assert(num_cu == 120);
        }
        else if(arch == "gfx1030")
        {
            assert(num_cu == 72 || num_cu == 36);
        }
        else if(arch == "gfx90a")
        {
            assert(num_cu == 110 || num_cu == 104);
        }
        else
            throw std::runtime_error("Invalid Arch Name");
    }

    void PrepBatchNorm()
    {
        VerifyDevProps();
        command         = job["config"];
        command["bias"] = 0;
        SetBNDescriptor();
        is_fwd_train = (job["direction"].get<int>() == 0 || job["direction"].get<int>() & 1);
        is_fwd_infer = (job["direction"].get<int>() == 0 || job["direction"].get<int>() & 2);
        is_bwd       = (job["direction"].get<int>() == 0 || job["direction"].get<int>() & 4);
    }

    // Getters and setters
    std::vector<int> GetInputTensorLengths();
    std::vector<int> GetBiasTensorLengths();
    int SetBNDescriptor();
    miopen::conv::Direction GetDirection() const;

    int ProcessStep(const std::string& step_name) override;

    // Steps
    int TestApplicability();
    int GetandSetData();
    miopen::batchnorm::ProblemDescription GetProblemDescription();
    std::vector<miopen::solver::ConvSolution> GetBNSolutions(miopen::ExecutionContext& ctx);

    // Utility functions
    void InitNoGpuHandle(miopen::Handle& handle);
    auto GetFwdTrainSolvers();
    auto GetFwdInferSolvers();
    auto GetBwdSolvers();

    json command;
    json job;

    miopenBatchNormMode_t bn_mode;
    std::vector<std::string> steps_processed;
    bool saveMeanVar;
    bool keepRunningMeanVar;

    double epsilon        = 1.0;
    double expAvgFactor   = 1.0;
    bool isDepthSpecified = false;
    int forw              = 0;
    int back              = 1;
    bool is_fwd_train     = true;
    bool is_fwd_infer     = false;
    bool is_bwd           = false;

    tensor<Tgpu, Tcpu> inputTensor;
    tensor<Tgpu, Tcpu> outputTensor;
    tensor<Tgpu, Tcpu> biasScaleTensor;

    // for backward
    tensor<Tgpu, Tcpu> dyInputTensor;
    tensor<Tgpu, Tcpu> dxOutputTensor;
};

template <typename Tgpu, typename Tref>
int BNFin<Tgpu, Tref>::TestApplicability()
{
#if MIOPEN_MODE_NOGPU
    GetandSetData();
#else
    throw std::runtime_error("MIOpen needs to be compiled with the NOGPU backend "
                             "to test applicability");
#endif

    auto& handle = GetHandle();
    auto ctx     = miopen::ExecutionContext(&handle);
#if MIOPEN_MODE_NOGPU
    InitNoGpuHandle(handle);
#else
    throw std::runtime_error("MIOpen needs to be compiled with the NOGPU backend "
                             "to test applicability");
#endif
    ctx.SetStream(&handle);
    ctx.DetectRocm();

    std::vector<std::string> app_solvers;

    const auto slns = GetBNSolutions(ctx);
    for(auto it = slns.begin(); it != slns.end(); ++it)
    {
        std::cout << it->solver_id << std::endl;
        if(!it->invoker_factory)
        {
            MIOPEN_THROW(miopenStatusInternalError, "Invoker missing in solver " + it->solver_id);
        }
        app_solvers.push_back(it->solver_id);
    }
    for(auto& elem : app_solvers)
    {
        std::cout << elem << std::endl;
    }

    output["applicable_solvers"] = app_solvers;
    return 0;
}

template <typename Tgpu, typename Tref>
int BNFin<Tgpu, Tref>::GetandSetData()
{

    SetBNDescriptor();

    auto in_len = GetInputTensorLengths();

    if(command["bias"].get<int>() != 0)
    {
        auto bias_len   = GetBiasTensorLengths();
        biasScaleTensor = {GetHandle().GetStream(), bias_len, true, true};
    }

    std::vector<int> sb_len;
    if(bn_mode == miopenBNPerActivation)
    {
        // 1xCxHxW | in_len.size = 4
        sb_len.push_back(1);
        sb_len.push_back(in_len[1]);
        sb_len.push_back(in_len[2]);
        sb_len.push_back(in_len[3]);

        // 1xCxDxHxW | in_len.size = 5
        if(in_len.size() == 5)
        {
            sb_len.push_back(in_len[4]);
        }
    }
    else if(bn_mode == miopenBNSpatial)
    { // 1xCx1x1
        sb_len.push_back(1);
        sb_len.push_back(in_len[1]);
        sb_len.push_back(1);
        sb_len.push_back(1);

        // 1xCx1x1x1
        if(in_len.size() == 5)
        {
            sb_len.push_back(1);
        }
    }

    miopenSetTensorDescriptor(&inputTensor.desc, data_type, in_len.size(), in_len.data(), nullptr);
    miopenSetTensorDescriptor(
        &biasScaleTensor.desc, data_type, sb_len.size(), sb_len.data(), nullptr);
    miopenSetTensorDescriptor(&outputTensor.desc, data_type, in_len.size(), in_len.data(), nullptr);

    // backwards
    miopenSetTensorDescriptor(
        &dyInputTensor.desc, data_type, in_len.size(), in_len.data(), nullptr);
    miopenSetTensorDescriptor(
        &dxOutputTensor.desc, data_type, in_len.size(), in_len.data(), nullptr);
    return (0);
}

template <typename Tgpu, typename Tref>
std::vector<int> BNFin<Tgpu, Tref>::GetInputTensorLengths()
{
    int in_n = command["batchsize"];
    int in_c = command["in_channels"];
    int in_h = command["in_h"];
    int in_w = command["in_w"];
    int in_d = command["in_d"];

    if(command["in_d"] > 1)
    {
        isDepthSpecified = true;
        // NxCxDxHxW -> NxCx(D*H)xW
        return std::vector<int>({in_n, in_c, in_d, in_h, in_w});
    }
    else
    {
        isDepthSpecified = false;
        return std::vector<int>({in_n, in_c, in_h, in_w});
    }
}

template <typename Tgpu, typename Tref>
std::vector<int> BNFin<Tgpu, Tref>::GetBiasTensorLengths()
{
    int spatial_dim = 2;
    if(command["in_d"] > 1)
    {
        spatial_dim = 3;
    }

    std::vector<int> bias_lens(2 + spatial_dim, 1);

    bias_lens[1] = command["out_channels"];

    return bias_lens;
}

template <typename Tgpu, typename Tref>
int BNFin<Tgpu, Tref>::ProcessStep(const std::string& step_name)
{
    steps_processed.push_back(step_name);
    if(step_name == "applicability")
    {
        return TestApplicability();
    }
    return 0;
}

template <typename Tgpu, typename Tref>
int BNFin<Tgpu, Tref>::SetBNDescriptor()
{
    //    	double bnAlpha = inflags.GetValueDouble("alpha");
    //    	double bnBeta = inflags.GetValueDouble("beta");

    // batch norm mode type
    if(command["mode"] == 0)
    {
        bn_mode = miopenBNPerActivation;
    }
    else if(command["mode"] == 1)
    {
        bn_mode = miopenBNSpatial;
    }

    // save off mean and variance?
    if(command["save"] == 0)
    {
        saveMeanVar = false;
    }
    else if(command["save"] == 1)
    {
        saveMeanVar = true;
    }

    // keep running mean and variance
    if(command["run"] == 0)
    {
        keepRunningMeanVar = false;
    }
    else if(command["run"] == 1)
    {
        keepRunningMeanVar = true;
    }

    forw = command["forw"];
    back = command["back"];

    epsilon = 1;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
void BNFin<Tgpu, Tref>::InitNoGpuHandle(miopen::Handle& handle)
{
#if MIOPEN_MODE_NOGPU
    handle.impl->device_name        = job["arch"];
    handle.impl->num_cu             = job["num_cu"];
    handle.impl->max_mem_alloc_size = 32UL * 1024 * 1024 * 1024; // 32 GB
    handle.impl->global_mem_size    = 32UL * 1024 * 1024 * 1024;
    handle.impl->target_properties.Init(&handle);
#else
    std::ignore = handle;
#endif
}

template <typename Tgpu, typename Tref>
auto BNFin<Tgpu, Tref>::GetFwdTrainSolvers()
{
    return miopen::solver::SolverContainer<miopen::solver::batchnorm::BnFwdTrainingSpatialSingle,
                                           miopen::solver::batchnorm::BnFwdTrainingSpatialMultiple,
                                           miopen::solver::batchnorm::BnFwdTrainingPerActivation>{};
}

template <typename Tgpu, typename Tref>
auto BNFin<Tgpu, Tref>::GetFwdInferSolvers()
{
    return miopen::solver::SolverContainer<miopen::solver::batchnorm::BnFwdInference>{};
}

template <typename Tgpu, typename Tref>
auto BNFin<Tgpu, Tref>::GetBwdSolvers()
{
    return miopen::solver::SolverContainer<miopen::solver::batchnorm::BnBwdTrainingSpatialSingle,
                                           miopen::solver::batchnorm::BnBwdTrainingSpatialMultiple,
                                           miopen::solver::batchnorm::BnBwdTrainingPerActivation>{};
}

template <typename Tgpu, typename Tref>
miopen::batchnorm::ProblemDescription BNFin<Tgpu, Tref>::GetProblemDescription()
{
    if(is_fwd_train)
    {
        return miopen::batchnorm::ProblemDescription{bn_mode,
                                                     inputTensor.desc,
                                                     outputTensor.desc,
                                                     biasScaleTensor.desc,
                                                     expAvgFactor,
                                                     epsilon,
                                                     saveMeanVar,
                                                     keepRunningMeanVar};
    }
    else if(is_fwd_infer)
    {
        return miopen::batchnorm::ProblemDescription(
            bn_mode, inputTensor.desc, outputTensor.desc, biasScaleTensor.desc, epsilon);
    }
    else if(is_bwd)
    {
        return miopen::batchnorm::ProblemDescription(bn_mode,
                                                     inputTensor.desc,
                                                     dyInputTensor.desc,
                                                     dxOutputTensor.desc,
                                                     biasScaleTensor.desc,
                                                     epsilon,
                                                     saveMeanVar);
    }
    else
    {
        throw std::runtime_error("Unable to determine batch norm direction");
    }
}

template <typename Tgpu, typename Tref>
std::vector<miopen::solver::ConvSolution>
BNFin<Tgpu, Tref>::GetBNSolutions(miopen::ExecutionContext& ctx)
{
    const auto problem = GetProblemDescription();
    if(is_fwd_train)
    {
        return GetFwdTrainSolvers().SearchForSolutions(ctx, problem, 1);
    }
    else if(is_fwd_infer)
    {
        return GetFwdInferSolvers().SearchForSolutions(ctx, problem, 1);
    }
    else if(is_bwd)
    {
        return GetBwdSolvers().SearchForSolutions(ctx, problem, 1);
    }
    else
    {
        throw std::runtime_error("Unable to determine batch norm direction");
    }
}

} // namespace fin
#endif // GUARD_MIOPEN_BN_FIN_HPP
