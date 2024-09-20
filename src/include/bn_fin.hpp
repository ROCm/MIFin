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
#include <miopen/filesystem.hpp>
#include <miopen/miopen.h>
#include <miopen/batchnorm/problem_description.hpp>
#include <miopen/batch_norm.hpp>
#include <miopen/batchnorm/invoke_params.hpp>
#include <miopen/batchnorm/solvers.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/solver.hpp>
#include <miopen/solver_id.hpp>
#include <miopen/driver_arguments.hpp>

#include <nlohmann/json.hpp>

#define EPSILON 1e-3

namespace fs = miopen::fs;

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
    miopen::debug::BatchNormDirection_t GetDirection() const;

    int ProcessStep(const std::string& step_name) override;

    // Steps
    int TestApplicability();
    int GetandSetData();
    std::vector<miopen::solver::ConvSolution> GetBNSolutions(miopen::ExecutionContext& ctx);
    miopen::batchnorm::ProblemDescription GetProblemDescription();
    auto GetAlgorithm();

    int MIOpenCompile(TuningOp tuning_op);

    float PerfTune(const miopen::Handle& h,
                   const miopen::batchnorm::ProblemDescription& problem,
                   const miopen::solver::Id& solver_id,
                   miopen::PerformanceDb& db,
                   miopen::ExecutionContext& perf_ctx);

    float FindTune(const miopen::Handle& h, const miopen::solver::ConvSolution& solution);
    int MIOpenEval(TuningOp tuning_op);

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
    bool is_fwd_train       = true;
    bool is_fwd_infer       = false;
    bool is_bwd             = false;

    tensor<Tgpu, Tcpu> inputTensor;
    tensor<Tgpu, Tcpu> outputTensor;
    tensor<Tgpu, Tcpu> biasScaleTensor;
    tensor<Tgpu, Tcpu> workspace;

    // for backward
    tensor<Tgpu, Tcpu> dyInputTensor;
    tensor<Tgpu, Tcpu> dxOutputTensor;
};

template <typename Tgpu, typename Tref>
miopen::debug::BatchNormDirection_t BNFin<Tgpu, Tref>::GetDirection() const
{
    return is_fwd_train ? miopen::debug::BatchNormDirection_t::ForwardTraining
                        : (is_fwd_infer ? miopen::debug::BatchNormDirection_t::ForwardInference
                                        : miopen::debug::BatchNormDirection_t::Backward);
}
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
    // cppcheck-suppress unreadVariable
    auto ctx = miopen::ExecutionContext(&handle);
#if MIOPEN_MODE_NOGPU
    BaseFin::InitNoGpuHandle(handle, job["arch"], job["num_cu"]);
#else
    throw std::runtime_error("MIOpen needs to be compiled with the NOGPU backend "
                             "to test applicability");
#endif
    ctx.SetStream(&handle);

    std::vector<std::string> app_solvers;

    for(const auto& sln : GetBNSolutions(ctx))
    {
        std::cerr << sln.solver_id << std::endl;
        if(!sln.invoker_factory)
        {
            MIOPEN_THROW(miopenStatusInternalError, "Invoker missing in solver " + sln.solver_id);
        }
        app_solvers.push_back(sln.solver_id);
    }
    for(const auto& elem : app_solvers)
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
        biasScaleTensor = {GetHandle().GetStream(), GetBiasTensorLengths(), true, true};
    }
    else
    {
        biasScaleTensor = {GetHandle().GetStream(), sb_len, true, true};
    }

    // sanity check for memory layout
    if(GetMemLayout(command["in_layout"]) != miopenTensorLayout_t::miopenTensorNCHW)
        throw std::runtime_error("Provided memory layout is :" + std::string(command["in_layout"]) +
                                 ". Batch norm only support default NCHW");
    if(GetMemLayout(command["in_layout"]) != miopenTensorLayout_t::miopenTensorNCHW)
        throw std::runtime_error(
            "Provided memory layout is : " + std::string(command["in_layout"]) +
            ". Batch norm only support default NCHW");

    inputTensor  = {GetHandle().GetStream(), in_len, true, false};
    outputTensor = {GetHandle().GetStream(), in_len, false, true};

    // backwards
    dyInputTensor  = {GetHandle().GetStream(), in_len, false, true};
    dxOutputTensor = {GetHandle().GetStream(), in_len, true, false};
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
        return TestApplicability();
    if(step_name == "miopen_perf_compile")
        return MIOpenCompile(TuningOp::Perf);
    if(step_name == "miopen_perf_eval")
        return MIOpenEval(TuningOp::Perf);
    if(step_name == "miopen_find_compile")
        return MIOpenCompile(TuningOp::Find);
    if(step_name == "miopen_find_eval")
        return MIOpenEval(TuningOp::Find);
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
        throw std::runtime_error("Unable to get solvers for batch norm");
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
        throw std::runtime_error("Unable to get solvers for batch norm");
    }
}

template <typename Tgpu, typename Tref>
int BNFin<Tgpu, Tref>::MIOpenCompile(TuningOp tuning_op)
{
    std::cerr << "MIOpenFinCompile" << std::endl;
    std::cerr << "Processing command: " << command << std::endl;
#if MIOPEN_MODE_NOGPU
    GetandSetData();
#else
    throw std::runtime_error(
        "Unable to perform MIOpenCompile MIOpen was not compiled using HIPNOGPU backend");
#endif
    auto& handle = GetHandle();
    // cppcheck-suppress unreadVariable
    auto ctx = miopen::ExecutionContext(&handle);
    GetHandle().EnableProfiling(true);
#if MIOPEN_MODE_NOGPU
    BaseFin::InitNoGpuHandle(handle, job["arch"], job["num_cu"]);
#else
    throw std::runtime_error("MIOpen needs to be compiled with the NOGPU backend "
                             "for Batch Norm find_compile");
#endif
    ctx.SetStream(&handle);

    const auto problem         = GetProblemDescription();
    const auto network_config  = problem.MakeNetworkConfig();
    output["network_config"]   = network_config;
    output["db_key"]           = network_config.ToString();
    output["is_winograd_only"] = false;

    json find_result;
    std::cerr << "Job Arch: " << job["arch"]
              << ": Handle Arch: " << handle.GetTargetProperties().Name() << std::endl;
    std::cerr << "Job Num CU: " << job["num_cu"]
              << ": Handle Num Cu: " << handle.GetMaxComputeUnits() << std::endl;

    std::vector<miopen::solver::Id> solver_list;
    if(job.contains("solvers"))
        for(std::string solver_str : job["solvers"]) // cppcheck-suppress useStlAlgorithm
            solver_list.push_back(miopen::solver::Id(solver_str));
    else
        solver_list = miopen::solver::GetSolversByPrimitive(miopen::solver::Primitive::Batchnorm);

    if(job.contains("dynamic_only"))
        ctx.use_dynamic_solutions_only = true;

    auto db = GetDb(ctx);
    json comp_res;

    for(const auto& sln : GetBNSolutions(ctx))
    {
        json res_item;
        res_item["reason"]    = "";
        auto process_solution = [&]() -> bool {
            // remove the user db files
            fs::remove_all(miopen::GetCachePath(false));
            std::cerr << "Processing Solver: " << sln.solver_id << std::endl;
            // const auto& s           = sln.GetSolver();
            res_item["solver_name"] = sln.solver_id;
            res_item["algorithm"]   = GetAlgorithm();

            if(tuning_op == TuningOp::Perf)
            {
                std::vector<miopen::solver::KernelInfo> kernels;
                for(auto&& kernel : sln.construction_params) // cppcheck-suppress useStlAlgorithm
                    kernels.push_back(kernel);
                std::ignore = miopen::solver::PrecompileKernels(handle, kernels);

                res_item["kernel_objects"] = BuildJsonKernelList(handle, kernels);
            }
            else if(tuning_op == TuningOp::Find)
            {
                //  NOTE: how to get params from solution?
                //  res_item["params"]    = ???s.GetPerfCfgParams(ctx, problem, db);
                res_item["workspace"]      = sln.workspace_sz;
                res_item["kernel_objects"] = BuildJsonKernelList(handle, sln.construction_params);
            }
            res_item["tunable"] = true;
            res_item["reason"]  = "Success";
            return true;
        };

        auto res = process_solution();

        if(tuning_op == TuningOp::Perf)
            res_item["perf_compiled"] = res;
        if(tuning_op == TuningOp::Find)
            res_item["find_compiled"] = res;
        comp_res.push_back(res_item);
    }
    if(tuning_op == TuningOp::Perf)
        output["miopen_perf_compile_result"] = comp_res;
    if(tuning_op == TuningOp::Find)
        output["miopen_find_compile_result"] = comp_res;
    return 1;
}

template <typename Tgpu, typename Tref>
int BNFin<Tgpu, Tref>::MIOpenEval(TuningOp tuning_op)
{
    std::cerr << "MIOpenEval" << std::endl;
    std::cerr << "Processing command: " << command << std::endl;
// Before this step is executed, the following steps should have been evaluated
// alloc_buf only if only timing is required
// alloc_buf, fill_buf and copy_buf_to_device if numerical accuracy would be
// checked ??
#if MIOPEN_MODE_NOGPU
    throw std::runtime_error("Unable to run MIOpenEval, Invalid MIOpen backend: HIPNOGPU");
#endif
    const auto conv_dir = GetDirection();
    // The first arg to the DataInvokeParams changes based on direction
    const auto problem = GetProblemDescription();

    GetHandle().EnableProfiling(true);
    auto& h = GetHandle();
    // cppcheck-suppress unreadVariable
    auto ctx = miopen::ExecutionContext(&h);
    ctx.SetStream(&(h));
    // problem.SetupFloats(ctx);

    output["is_winograd_only"] = false;
    const auto network_config  = problem.MakeNetworkConfig();
    output["network_config"]   = network_config;
    std::ostringstream ss;
    // problem.Serialize(ss);
    output["db_key"] = ss.str();

    auto db = GetDb(ctx, problem);
    json eval_result;
    const auto& tgt_props  = h.GetTargetProperties();
    const std::string arch = tgt_props.Name();
    const size_t num_cu    = h.GetMaxComputeUnits();
    std::cerr << "Job Arch: " << job["arch"] << ": Handle Arch: " << arch << std::endl;
    std::cerr << "Job Num CU: " << job["num_cu"] << ": Handle Num Cu: " << num_cu << std::endl;
    bool dynamic_only = false;
    if(job.contains("dynamic_only"))
        ctx.use_dynamic_solutions_only = true;

    std::string comp_res_str;
    if(tuning_op == TuningOp::Perf)
        comp_res_str = "miopen_perf_compile_result";
    else if(tuning_op == TuningOp::Find)
        comp_res_str = "miopen_find_compile_result";

    for(const auto& eval_slv : job[comp_res_str])
    {
        json res_item;
        std::error_code ec;
        fs::remove_all(miopen::GetCachePath(false), ec);
        // fs::remove_all(miopen::GetCachePath(true), ec);
        if(ec)
        {
            std::cerr << "Error while removing MIOpen cache: " << ec.message();
        }
        auto process_solver = [&]() -> bool {
            const std::string solver_name = eval_slv["solver_name"];
            std::cerr << "Processing solver: " << solver_name << std::endl;
            const auto solver_id    = miopen::solver::Id{solver_name};
            const auto& s           = solver_id.GetSolver();
            res_item["solver_name"] = solver_name;
            res_item["algorithm"]   = GetAlgorithm();

            if(s.IsEmpty())
            {
                res_item["reason"] = "Empty Solver";
                std::cerr << "Skipping invalid solver: " << solver_id.ToString() << std::endl;
                return false;
            }
            if(dynamic_only && !s.IsDynamic())
            {
                res_item["reason"] = "Not Dynamic";
                std::cerr << "Skipping static solver: " << solver_id.ToString() << std::endl;
                return false;
            }

            // Get the binary
            std::cerr << "Applicable solver: " << solver_name << ", loading binaries from fin input"
                      << std::endl;
            if(!LoadJsonKernelList(h, eval_slv["kernel_objects"], res_item))
                return false;

            // auto solution = s.FindSolution(ctx, problem, db, {}); // auto tune is not expected
            auto solution = FindSolution(s, ctx, problem, db, ctx, "", std::nullopt);
            if(!solution.Succeeded())
                std::cerr << "Applicable solver did not succeeded" << std::endl;
            SolutionHasProgram(h, solution);

            std::cerr << "Checking workspace size" << std::endl;
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
                float kernel_time = -1;
                if(tuning_op == TuningOp::Perf)
                    kernel_time = PerfTune(h, problem, solver_id, db, ctx);
                else if(tuning_op == TuningOp::Find)
                    kernel_time = FindTune(h, solution);

                json kern_objs = BuildJsonKernelList(h, solution.construction_params);

                res_item["tunable"] = s.IsTunable();
                // ??res_item["params"     = s.GetPerfCfgParams(ctx, problem, db);
                res_item["workspace"] = solution.workspace_sz;
                res_item["time"]      = kernel_time;
                // ?res_item["layout"]     = problem.GetInLayout();
                // ?res_item["data_type"]  = problem.GetInDataType();
                res_item["direction"] = conv_dir;
                // ??res_item["bias"]      = problem.GetBias();
                res_item["kernel_objects"] = kern_objs;
                res_item["reason"]         = "Success";
                if(kernel_time == 0.0)
                    res_item["reason"] = "Invoker returned time = 0";
                if(kernel_time < 0)
                    res_item["reson"] = "kernel_time not measured";
            }
            catch(const std::exception& e)
            {
                res_item["reason"] = std::string("Invoker exception: ") + e.what();
                std::cerr << res_item["reason"] << std::endl;
                return false;
            }

            return true;
        };
        auto res = process_solver();
    }
    return true;
}

float BNFin<Tgpu, Tref>::PerfTune(const miopen::Handle& h,
                                  const miopen::conv::ProblemDescription& problem,
                                  const miopen::solver::Id& solver_id,
                                  miopen::PerformanceDb& db,
                                  miopen::ExecutionContext& perf_ctx)
{
    return true;
}

template <typename Tgpu, typename Tref>
float ConvFin<Tgpu, Tref>::FindTune(const miopen::Handle& h,
                                    const miopen::solver::ConvSolution& solution)
{
    return true;
}

} // namespace fin
#endif // GUARD_MIOPEN_BN_FIN_HPP
