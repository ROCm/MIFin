/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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
template <typename Tgpu, typename Tcpu>
class BNFin : public BaseFin
{
    public:
    BNFin() : BaseFin() {}
    BNFin(json _job) : BaseFin(), job(_job)
    {
        if(job.contains("config"))
            PrepBatchNorm();
    }

    void PrepBatchNorm()
    {
        BaseFin::VerifyDevProps(job["arch"], job["num_cu"]);
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
    auto GetFwdTrainSolvers();
    auto GetFwdInferSolvers();
    auto GetBwdSolvers();

    json command;
    json job;

    miopenBatchNormMode_t bn_mode;
    std::vector<std::string> steps_processed;
    bool saveMeanVar        = false;
    bool keepRunningMeanVar = false;
    double epsilon          = 1.0;
    double expAvgFactor     = 1.0;
    bool isDepthSpecified   = false;
    int forw                = 0;
    int back                = 1;
    bool is_fwd_train       = true;
    bool is_fwd_infer       = false;
    bool is_bwd             = false;

    miopen::TensorDescriptor inputTensor;
    miopen::TensorDescriptor outputTensor;
    miopen::TensorDescriptor biasScaleTensor;

    // for backward
    miopen::TensorDescriptor dyInputTensor;
    miopen::TensorDescriptor dxOutputTensor;
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
    BaseFin::InitNoGpuHandle(handle, job["arch"], job["num_cu"]);
#else
    throw std::runtime_error("MIOpen needs to be compiled with the NOGPU backend "
                             "to test applicability");
#endif
    ctx.SetStream(&handle);
    ctx.DetectRocm();

    std::vector<std::string> app_solvers;

    for(const auto& sln : GetBNSolutions(ctx))
    {
        std::cout << sln.solver_id << std::endl;
        if(!sln.invoker_factory)
        {
            MIOPEN_THROW(miopenStatusInternalError, "Invoker missing in solver " + sln.solver_id);
        }
        app_solvers.push_back(sln.solver_id);
    }
    for(auto& elem : app_solvers)
    {
        std::cerr << elem << std::endl;
    }

    output["applicable_solvers"] = app_solvers;
    return 0;
}

template <typename Tgpu, typename Tref>
int BNFin<Tgpu, Tref>::GetandSetData()
{

    SetBNDescriptor();

    auto in_len = GetInputTensorLengths();

    std::vector<int> sb_len;
    if(bn_mode == miopenBNPerActivation)
    {
        // 1xCxHxW | in_len.size = 4
        sb_len = {1, in_len[1], in_len[2], in_len[3]};

        // 1xCxDxHxW | in_len.size = 5
        if(in_len.size() == 5)
        {
            sb_len.push_back(in_len[4]);
        }
    }
    else if(bn_mode == miopenBNSpatial)
    { // 1xCx1x1
        sb_len = {1, in_len[1], 1, 1};

        // 1xCx1x1x1
        if(in_len.size() == 5)
        {
            sb_len.push_back(1);
        }
    }

    if(command["bias"].get<int>() != 0)
    {
        biasScaleTensor = miopen::TensorDescriptor(data_type,  GetBiasTensorLengths());
    }
    else
    {
        biasScaleTensor = miopen::TensorDescriptor(data_type, sb_len.data(), sb_len.size());
    }

    inputTensor     = miopen::TensorDescriptor(data_type, in_len.data(), in_len.size());
    biasScaleTensor = miopen::TensorDescriptor(data_type, sb_len.data(), sb_len.size());
    outputTensor    = miopen::TensorDescriptor(data_type, in_len.data(), in_len.size());

    // backwards
    dyInputTensor  = miopen::TensorDescriptor(data_type, in_len.data(), in_len.size());
    dxOutputTensor = miopen::TensorDescriptor(data_type, in_len.data(), in_len.size());
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
    // batch norm mode type
    bn_mode = command["mode"] == 0 ? miopenBNPerActivation : miopenBNSpatial;

    // save off mean and variance?
    saveMeanVar = command["save"] == 0 ? false : true;

    // keep running mean and variance
    keepRunningMeanVar = command["run"] == 0 ? false : true;

    forw = command["forw"];
    back = command["back"];

    epsilon = 1;

    return miopenStatusSuccess;
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
                                                     inputTensor,
                                                     outputTensor,
                                                     biasScaleTensor,
                                                     expAvgFactor,
                                                     epsilon,
                                                     saveMeanVar,
                                                     keepRunningMeanVar};
    }
    else if(is_fwd_infer)
    {
        return miopen::batchnorm::ProblemDescription(
            bn_mode, inputTensor, outputTensor, biasScaleTensor, epsilon);
    }
    else if(is_bwd)
    {
        return miopen::batchnorm::ProblemDescription(bn_mode,
                                                     inputTensor,
                                                     dyInputTensor,
                                                     dxOutputTensor,
                                                     biasScaleTensor,
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
