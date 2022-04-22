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
#include <miopen/find_solution.hpp>
#include <miopen/solver.hpp>

#include <nlohmann/json.hpp>

#define EPSILON 1e-3

namespace fin {

using json = nlohmann::json;
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
    miopen::batchnorm::Algorithm GetAlgorithm();
    int MIOpenFindCompile();
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

    // tensor<Tgpu, Tcpu> inputTensor;
    // tensor<Tgpu, Tcpu> outputTensor;
    // tensor<Tgpu, Tcpu> biasScaleTensor;
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
    fin::InitNoGpuHandle(handle, job["arch"], job["num_cu"]);
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
        auto bias_len = GetBiasTensorLengths();
        // biasScaleTensor = {GetHandle().GetStream(), bias_len, true, true};
        // biasScaleTensor = miopen::TensorDescriptor(data_type, sb_len.data(), bias_len.size());
    }

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
        auto bias_len = GetBiasTensorLengths();
        // biasScaleTensor = {GetHandle().GetStream(), bias_len, true, true};
        biasScaleTensor = miopen::TensorDescriptor(data_type, sb_len.data(), bias_len.size());
    }
    else
    {
        biasScaleTensor = miopen::TensorDescriptor(data_type, sb_len.data(), sb_len.size());
    }

    // miopenSetTensorDescriptor(&inputTensor.desc, data_type, in_len.size(), in_len.data(),
    // nullptr);
    inputTensor = miopen::TensorDescriptor(data_type, in_len.data(), in_len.size());

    // miopenSetTensorDescriptor(
    //    &biasScaleTensor.desc, data_type, sb_len.size(), sb_len.data(), nullptr);
    biasScaleTensor = miopen::TensorDescriptor(data_type, sb_len.data(), sb_len.size());

    // miopenSetTensorDescriptor(&outputTensor.desc, data_type, in_len.size(), in_len.data(),
    // nullptr);
    outputTensor = miopen::TensorDescriptor(data_type, in_len.data(), in_len.size());

    // backwards
    // miopenSetTensorDescriptor(
    //    &dyInputTensor.desc, data_type, in_len.size(), in_len.data(), nullptr);
    dyInputTensor = miopen::TensorDescriptor(data_type, in_len.data(), in_len.size());

    // miopenSetTensorDescriptor(
    //    &dxOutputTensor.desc, data_type, in_len.size(), in_len.data(), nullptr);
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
    if(step_name == "miopen_find_compile")
    {
        return MIOpenFindCompile();
    }
    return 0;
}

template <typename Tgpu, typename Tref>
int BNFin<Tgpu, Tref>::SetBNDescriptor()
{
    //    	double bnAlpha = inflags.GetValueDouble("alpha");
    //    	double bnBeta = inflags.GetValueDouble("beta");

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
        throw std::runtime_error("Unable to get sovlers for batch norm");
    }
}

template <typename Tgpu, typename Tref>
miopen::batchnorm::Algorithm BNFin<Tgpu, Tref>::GetAlgorithm()
{
    if(is_fwd_train)
    {
        return bn_mode == miopenBNSpatial
                          ? AlgorithmName{"miopenBatchNormForwardTrainingSpatial"}
                          : AlgorithmName{"miopenBatchNormForwardTrainingPerActivation"};
    }
    else if(is_fwd_infer)
    {
        return AlgorithmName{"miopenBatchNormalizationForwardInference"};
    }
    else if(is_bwd)
    {
        return bn_mode == miopenBNSpatial
                          ? AlgorithmName{"miopenBatchNormBackwardPropSpatial"}
                          : AlgorithmName{"miopenBatchNormBackwardPropPerActivation"};
    }
    else
    {
        throw std::runtime_error("Unable to get sovlers for batch norm");
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
        throw std::runtime_error("Unable to to get solutions for batch norm");
    }
}



template <typename Tgpu, typename Tref>
int BNFin<Tgpu, Tref>::MIOpenFindCompile()
{
    std::cout << "MIOpenFinCompile" << std::endl;
    std::cout << "Processing command: " << command << std::endl;
#if MIOPEN_MODE_NOGPU
    GetandSetData();
#else
    throw std::runtime_error(
        "Unable to perform MIOpenFindCompile MIOpen was not compiled using HIPNOGPU backend");
#endif
    auto& handle = GetHandle();
    auto ctx     = miopen::ExecutionContext(&handle);
    GetHandle().EnableProfiling(true);
#if MIOPEN_MODE_NOGPU
    fin::InitNoGpuHandle(handle, job["arch"], job["num_cu"]);
#else
    throw std::runtime_error("MIOpen needs to be compiled with the NOGPU backend "
                             "for MIOpenFindCompile");
#endif
    ctx.SetStream(&handle);
    ctx.DetectRocm();
    // ctx.SetupFloats();

    // const auto network_config   = ctx.BuildConfKey();
    // const bool is_winograd_only = convDesc.IsWinograd3x3SupportedAndFast(ctx);
    // output["is_winograd_only"]  = is_winograd_only;
    // output["network_config"]    = network_config;
    std::ostringstream ss;
    const auto problem        = GetProblemDescription();
    const auto network_config = problem.MakeNetworkConfig();
    // problem.Serialize(ss);
    output["db_key"] = ss.str();
    /*const auto solver_list =
        miopen::solver::GetSolversByPrimitive(miopen::solver::Primitive::Batchnorm);
    for(const auto& solver_id : solver_list)
    {
        std::cerr << "'" << solver_id.ToString() << "'" << std::endl;
    }*/

    json find_result;
    const auto& tgt_props  = handle.GetTargetProperties();
    const std::string arch = tgt_props.Name();
    const size_t num_cu    = handle.GetMaxComputeUnits();
    std::cerr << "Job Arch: " << job["arch"] << ": Handle Arch: " << arch << std::endl;
    std::cerr << "Job Num CU: " << job["num_cu"] << ": Handle Num Cu: " << num_cu << std::endl;
    bool dynamic_only = false;
    if(job.contains("dynamic_only"))
        dynamic_only = job["dynamic_only"];
    const auto slns  = GetBNSolutions(ctx);

    for(auto it = slns.begin(); it != slns.end(); ++it)
    {
        // remove the user db files
        std::cout << it->solver_id << std::endl;
        boost::filesystem::remove_all(miopen::GetCachePath(false));
        json res_item;
        res_item["solver_id"] = it->solver_id;
        const auto solver  = miopen::solver::Id(it->solver_id);
        std::cout << solver.ToString() << std::endl;
        // const auto sid = miopen::solver::Id(it->solver_id);
        // const auto algo = sid.GetAlgo();
        const auto solver_list =
            miopen::solver::GetSolversByPrimitive(miopen::solver::Primitive::Batchnorm);
        for(const auto& solver_id : solver_list)
        {
            std::cerr << solver_id.ToString() << " - " << it->solver_id << std::endl;
        }

        res_item["reason"]    = "Success";
        res_item["workspace"] = it->workspace_sz;
        std::cout << "res_item" << res_item << std::endl;
        std::vector<miopen::solver::KernelInfo> kernels;
        for(auto&& kernel : it->construction_params)
        {
            kernels.push_back(kernel);
        }
        std::ignore      = miopen::solver::PrecompileKernels(handle, kernels);
        json kernel_list = json::array();
        for(const auto& k : kernels)
        {
            json kernel;
            auto comp_opts   = k.comp_options;
            auto p           = handle.LoadProgram(k.kernel_file, comp_opts, false, "");
            const auto hsaco = p.IsCodeObjectInMemory()
                                   ? p.GetCodeObjectBlob()
                                   : miopen::LoadFile(p.GetCodeObjectPathname().string());
            if(hsaco.empty())
            {
                std::cerr << "Got empty code object" << std::endl;
                throw std::runtime_error("Got empty code object");
            }
            // Compress the blob
            auto md5_sum             = miopen::md5(hsaco);
            auto size                = hsaco.size();
            bool success             = false;
            auto compressed_hsaco    = miopen::compress(hsaco, &success);
            const auto encoded_hsaco = base64_encode(compressed_hsaco);
            kernel["kernel_file"]    = k.kernel_file;
            kernel["comp_options"]   = k.comp_options;
            if(success)
            {
                kernel["uncompressed_size"] = size;
                kernel["md5_sum"]           = md5_sum;
                kernel["blob"]              = encoded_hsaco;
            }
            else
            {
                kernel["md5_sum"]           = "Failed to compress kernel";
                kernel["uncompressed_size"] = 0;
                kernel["blob"]              = "";
            }
            kernel_list.push_back(kernel);
            std::cerr << "Successfully added new kernel" << std::endl;
        }
        res_item["kernel_objects"] = kernel_list;
        res_item["find_compiled"]  = true;
        find_result.push_back(res_item);
    }
    output["miopen_find_compile_result"] = find_result;
    return 1;
}
} // namespace fin
#endif // GUARD_MIOPEN_BN_FIN_HPP
