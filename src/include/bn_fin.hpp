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
#include <miopen/solver_id.hpp>

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
    std::vector<miopen::solver::ConvSolution> GetBNSolutions(miopen::ExecutionContext& ctx);
    miopen::batchnorm::ProblemDescription GetProblemDescription();
    auto GetAlgorithm();
    int MIOpenFindCompile();
    int MIOpenFindEval();

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
    tensor<Tgpu, Tcpu> workspace;

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
        biasScaleTensor = miopen::TensorDescriptor(data_type, GetBiasTensorLengths());
    }
    else
    {
        biasScaleTensor = miopen::TensorDescriptor(data_type, sb_len.data(), sb_len.size());
    }

    inputTensor  = miopen::TensorDescriptor(data_type, in_len);
    outputTensor = miopen::TensorDescriptor(data_type, in_len);

    // backwards
    dyInputTensor  = miopen::TensorDescriptor(data_type, in_len);
    dxOutputTensor = miopen::TensorDescriptor(data_type, in_len);
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
    if(step_name == "miopen_find_eval")
    {
        return MIOpenFindEval();
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
auto BNFin<Tgpu, Tref>::GetAlgorithm()
{
    if(is_fwd_train)
    {
        return bn_mode == miopenBNSpatial
                   ? miopen::AlgorithmName{"miopenBatchNormForwardTrainingSpatial"}
                   : miopen::AlgorithmName{"miopenBatchNormForwardTrainingPerActivation"};
    }
    else if(is_fwd_infer)
    {
        return miopen::AlgorithmName{"miopenBatchNormalizationForwardInference"};
    }
    else if(is_bwd)
    {
        return bn_mode == miopenBNSpatial
                   ? miopen::AlgorithmName{"miopenBatchNormBackwardPropSpatial"}
                   : miopen::AlgorithmName{"miopenBatchNormBackwardPropPerActivation"};
    }
    else
    {
        throw std::runtime_error("Unable to get sovlers for batch norm");
    }
}

template <typename Tgpu, typename Tref>
int BNFin<Tgpu, Tref>::MIOpenFindCompile()
{
    std::cerr << "MIOpenFinCompile" << std::endl;
    std::cerr << "Processing command: " << command << std::endl;
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
    BaseFin::InitNoGpuHandle(handle, job["arch"], job["num_cu"]);
#else
    throw std::runtime_error("MIOpen needs to be compiled with the NOGPU backend "
                             "for MIOpenFindCompile");
#endif
    ctx.SetStream(&handle);
    ctx.DetectRocm();

    const auto problem         = GetProblemDescription();
    const auto network_config  = problem.MakeNetworkConfig();
    output["network_config"]   = network_config;
    output["db_key"]           = network_config.ToString();
    output["is_winograd_only"] = false;

    json find_result;
    std::cerr << "Job Arch: " << job["arch"] << ": Handle Arch: " << handle.GetMaxComputeUnits()
              << std::endl;
    std::cerr << "Job Num CU: " << job["num_cu"]
              << ": Handle Num Cu: " << handle.GetTargetProperties().Name() << std::endl;

    for(const auto& sln : GetBNSolutions(ctx))
    {
        // remove the user db files
        boost::filesystem::remove_all(miopen::GetCachePath(false));
        json res_item;
        res_item["solver_id"] = sln.solver_id;
        res_item["algorithm"] = GetAlgorithm();

        res_item["reason"]    = "Success";
        res_item["workspace"] = sln.workspace_sz;
        std::vector<miopen::solver::KernelInfo> kernels;
        for(auto&& kernel : sln.construction_params)
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

template <typename Tgpu, typename Tref>
int BNFin<Tgpu, Tref>::MIOpenFindEval()
{
    std::cout << "MIOpenFindEval" << std::endl;
    std::cout << "Processing command: " << command << std::endl;
#if MIOPEN_MODE_NOGPU
    GetandSetData();
#else
    throw std::runtime_error(
        "Unable to perform MIOpenFindEval MIOpen was not compiled using HIPNOGPU backend");
#endif
    auto& h = GetHandle();
    auto ctx     = miopen::ExecutionContext(&h);
    GetHandle().EnableProfiling(true);
#if MIOPEN_MODE_NOGPU
    BaseFin::InitNoGpuHandle(h, job["arch"], job["num_cu"]);
#else
    throw std::runtime_error("MIOpen needs to be compiled with the NOGPU backend "
                             "for MIOpenFindEval");
#endif
    ctx.SetStream(&h);
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
    const auto& tgt_props  = h.GetTargetProperties();
    const std::string arch = tgt_props.Name();
    const size_t num_cu    = h.GetMaxComputeUnits();
    std::cerr << "Job Arch: " << job["arch"] << ": Handle Arch: " << arch << std::endl;
    std::cerr << "Job Num CU: " << job["num_cu"] << ": Handle Num Cu: " << num_cu << std::endl;
    const auto slns  = GetBNSolutions(ctx);

    bool dynamic_only = false;
    if(job.contains("dynamic_only"))
        dynamic_only = job["dynamic_only"];

    for(const auto& kinder :
        job["miopen_find_compile_result"]) // The "miopen_find_compile_result" list generated
                                           // by miopen_find_compile operation
    {
        // Somehow the direction changes mid loop !
        json res_item;
        boost::system::error_code ec;
        boost::filesystem::remove_all(miopen::GetCachePath(false), ec);
        // boost::filesystem::remove_all(miopen::GetCachePath(true), ec);
        if(ec)
        {
            std::cerr << "Error while removing MIOpen cache: " << ec.message();
        }
        auto process_solver = [&]() -> bool {
            const std::string solver_name = kinder["solver_id"];
            std::cerr << "Processing solver: " << solver_name << std::endl;
            const auto solver_id    = miopen::solver::Id{solver_name};
            const auto& s           = solver_id.GetSolver();
            res_item["solver_name"] = solver_name;
            //const auto algo         = solver_id.GetAlgo(conv_dir);
            //res_item["algorithm"]   = algo;
            if(s.IsEmpty())
            {
                std::cerr << "Skipping invalid solver: " << solver_id.ToString() << std::endl;
                return false;
            }
            /*if(!s.IsApplicable(ctx))
            {
                std::cerr << "Solver inapplicable: " << solver_name << std::endl;
                throw std::runtime_error(
                    "InApplicable solver was sent to fin, check Tuna for errors");
                return false;
            }*/
            if(dynamic_only && !s.IsDynamic())
            {
                res_item["reason"] = "Not Dynamic";
                std::cerr << "Skipping static solver: " << solver_id.ToString() << std::endl;
                return false;
            }
            std::cerr << solver_name << " is applicable" << std::endl;
            const miopen::solver::ConvSolution solution;
            for(auto it = slns.begin(); it != slns.end(); ++it)
            {
                std::cout << it->solver_id << "-" << solver_name << std::endl;

                bool is_same = it->solver_id.compare(solver_name)==0;
                std::cout << is_same << std::endl;

            }
            //const auto solution   = s.FindSolution(ctx, db, {}); // auto tune is not expected here
            res_item["workspace"] = solution.workspace_sz;
            // Get the binary
            std::cerr << "loading binaries from fin input" << std::endl;
            for(const auto& kernel_obj : kinder["kernel_objects"])
            {
                const auto size          = kernel_obj["uncompressed_size"];
                const auto md5_sum       = kernel_obj["md5_sum"];
                const auto encoded_hsaco = kernel_obj["blob"];
                const auto decoded_hsaco = base64_decode(encoded_hsaco);
                const auto hsaco         = miopen::decompress(decoded_hsaco, size);
                std::string comp_opts    = kernel_obj["comp_options"];
                std::string kernel_file  = kernel_obj["kernel_file"];
                if(miopen::md5(hsaco) == md5_sum)
                {
                    auto p = miopen::Program{kernel_file, hsaco};
                    h.AddProgram(p, kernel_file, comp_opts);
                }
                else
                {
                    std::cerr << "Corrupt Binary Object" << std::endl;
                    throw std::runtime_error("Corrupt binary object");
                    return false;
                }
            }
            for(const auto& kern : solution.construction_params)
            {
                if(!h.HasProgram(kern.kernel_file, kern.comp_options))
                {
                    std::cerr << "Binary object check failed, either tuning params have changed or "
                                 "fin is unable to write binary to program cache"
                              << std::endl;
                }
            }
            std::cerr << "Checking for workspace" << std::endl;
            if(solution.workspace_sz > workspace.desc.GetNumBytes())
            {
                std::cerr << "Allocating " << solution.workspace_sz << " bytes for workspace"
                          << std::endl;
                workspace = tensor<Tgpu, Tref>{
                    q,
                    std::vector<size_t>{static_cast<size_t>(solution.workspace_sz / sizeof(Tgpu))},
                    false,
                    false};
                workspace.AllocateBuffers();
            }
            if(!solution.invoker_factory)
            {
                std::cerr << "Invoker not implemeted" << std::endl;
                res_item["reason"] = "Invoker not implemented";
                return false;
            }
            try
            {
                std::cerr << "Preparing invokers" << std::endl;
                const auto invoker =
                    h.PrepareInvoker(*solution.invoker_factory, solution.construction_params);
                // This is required because DataInvokeParams switches tensor order due to
                // direction and it does not have a
                // copy constructor or a default constructor
                std::cerr << "Finished preparing invokers" << std::endl;
                /*
                if(conv_dir == miopen::conv::Direction::Forward)
                {
                    const auto invoke_ctx =
                        miopen::conv::DataInvokeParams{{inputTensor.desc,
                                                        inputTensor.gpuData.buf.get(),
                                                        weightTensor.desc,
                                                        weightTensor.gpuData.buf.get(),
                                                        outputTensor.desc,
                                                        outputTensor.gpuData.buf.get()},
                                                       workspace.gpuData.buf.get(),
                                                       workspace.desc.GetNumBytes(),
                                                       convDesc.attribute.gfx90aFp16alt.GetFwd()};
                    for(auto idx = 0; idx < INVOKE_LIMIT; idx++)
                        invoker(h, invoke_ctx);
                }
                else if(conv_dir == miopen::conv::Direction::BackwardData)
                {
                    const auto invoke_ctx =
                        miopen::conv::DataInvokeParams{{outputTensor.desc,
                                                        outputTensor.gpuData.buf.get(),
                                                        weightTensor.desc,
                                                        weightTensor.gpuData.buf.get(),
                                                        inputTensor.desc,
                                                        inputTensor.gpuData.buf.get()},
                                                       workspace.gpuData.buf.get(),
                                                       workspace.desc.GetNumBytes(),
                                                       convDesc.attribute.gfx90aFp16alt.GetBwd()};
                    for(auto idx = 0; idx < INVOKE_LIMIT; idx++)
                        invoker(h, invoke_ctx);
                }
                else if(conv_dir == miopen::conv::Direction::BackwardWeights)
                {
                    const auto invoke_ctx =
                        miopen::conv::WrWInvokeParams{{outputTensor.desc,
                                                       outputTensor.gpuData.buf.get(),
                                                       inputTensor.desc,
                                                       inputTensor.gpuData.buf.get(),
                                                       weightTensor.desc,
                                                       weightTensor.gpuData.buf.get()},
                                                      workspace.gpuData.buf.get(),
                                                      workspace.desc.GetNumBytes(),
                                                      convDesc.attribute.gfx90aFp16alt.GetWrW()};
                    for(auto idx = 0; idx < INVOKE_LIMIT; idx++)
                        invoker(h, invoke_ctx);
                }
                else
                {
                    throw std::runtime_error("Invalid Direction");
                }*/
            }
            catch(const std::exception& e)
            {
                res_item["reason"] = std::string("Invoker exeception: ") + e.what();
                return false;
            }
            const auto time    = h.GetKernelTime();
            res_item["time"]   = time;
            res_item["reason"] = "Success";

            return true;
        };

        auto res              = process_solver();
        res_item["evaluated"] = res;
        find_result.push_back(res_item);
    }

    output["miopen_find_eval_result"] = find_result;
    return 1;
}
} // namespace fin
#endif // GUARD_MIOPEN_BN_FIN_HPP
