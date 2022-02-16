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

#include "base64.hpp"
#include "error.hpp"
#include "fin.hpp"
#include "random.hpp"
#include "tensor.hpp"

#include <miopen/algorithm.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/any_solver.hpp>
#include <miopen/binary_cache.hpp>
#include <miopen/bz2.hpp>
#include <miopen/conv/context.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/conv_solution.hpp>
#include <miopen/find_db.hpp>
#include <miopen/invoker.hpp>
#include <miopen/load_file.hpp>
#include <miopen/md5.hpp>
#include <miopen/perf_field.hpp>
#include <miopen/solver_id.hpp>
#include <miopen/find_solution.hpp> 

#include <miopen/batchnorm/problem_description.hpp>
#include <miopen/batch_norm.hpp>
#include <miopen/batchnorm/invoke_params.hpp>
#include <miopen/batchnorm/problem_description.hpp>
#include <miopen/batchnorm/solvers.hpp>

#if MIOPEN_MODE_NOGPU
#include <miopen/kernel_cache.hpp>
#include <miopen/nogpu/handle_impl.hpp>
#endif

#include <boost/range/adaptor/sliced.hpp>
#include <boost/filesystem.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <float.h>
#include <fstream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <numeric>
#include <sstream>
#include <type_traits>
#include <vector>

namespace fin {

using json             = nlohmann::json;
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
        std::cout << job["config"] << std::endl;
        command         = job["config"];
        command["bias"] = 0;
        // timing is always enabled
        is_fwd = (job["direction"].get<int>() == 0 || job["direction"].get<int>() & 1);
        is_bwd = (job["direction"].get<int>() == 0 || job["direction"].get<int>() & 2);
        SetBNDescriptor();
        // workspace_dev = nullptr; // TODO: replaced with a tensor class
        // the variable name is implementation dependent, checking size instead
    }


    // Getters and setters
    std::vector<int> GetInputTensorLengths();
    std::vector<int> GetBiasTensorLengths();
    int SetBNDescriptor();
    miopen::conv::Direction GetDirection() const;

    int ProcessStep(const std::string& step_name) override;

    // Steps
    int AllocateBuffers();
    int CalcWorkspace();
    int FillBuffers();
    int CopyToDevice();
    int CopyFromDevice();
    int RunGPU();
    int TestApplicability();
    int TestPerfDbValid();
    int GetandSetData();
    int GetSolverList();
    int MIOpenFind();

    // Utility functions
    bool IsInputTensorTransform() const;
    void InitNoGpuHandle(miopen::Handle& handle);
    json command;
    json job;

    //bool wrw_allowed = 0, bwd_allowed = 0, forward_allowed = 1;
    miopenBatchNormMode_t bn_mode;
    std::vector<std::string> steps_processed;
    bool saveMeanVar;
    bool bsaveMeanVar;
    bool keepRunningMeanVar;
    bool estimatedMeanVar;
    double epsilon;
    double expAvgFactor = 1.0; 
    bool isDepthSpecified = false;

    int forw = 0;
    int back = 1;
    bool is_fwd            = true;
    bool is_bwd            = false;

    //const miopen::TensorDescriptor inputTensor;
    //const miopen::TensorDescriptor biasScaleTensor;
    //const miopen::TensorDescriptor outputTensor;
    //miopenTensorDescriptor_t biasScaleTensor;
    //miopenTensorDescriptor_t outputTensor;
    tensor<Tgpu, Tcpu> inputTensor;
    tensor<Tgpu, Tcpu> outputTensor;
    tensor<Tgpu, Tcpu> biasScaleTensor;

    // Backwards
    miopenTensorDescriptor_t dyInputTensor;
    miopenTensorDescriptor_t dxOutputTensor;
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
    const auto problem = miopen::batchnorm::ProblemDescription{bn_mode,
                                                       inputTensor,
                                                       outputTensor,
                                                       biasScaleTensor,
                                                       expAvgFactor,
                                                       epsilon,
                                                       saveMeanVar,
                                                       keepRunningMeanVar};

    //auto ctx    = miopen::ConvolutionContext{problem};
    auto handle = miopen::Handle{};
    auto ctx = miopen::ExecutionContext(&handle);
#if MIOPEN_MODE_NOGPU
    InitNoGpuHandle(handle);
#else
    throw std::runtime_error("MIOpen needs to be compiled with the NOGPU backend "
                             "to test applicability");
#endif

    //ctx.SetStream(&handle);
    //ctx.DetectRocm();
    //ctx.SetupFloats();
    //const auto network_config = ctx.BuildConfKey();
    const auto solvers = miopen::solver::SolverContainer<miopen::solver::batchnorm::BnFwdTrainingSpatialSingle,
                                                 miopen::solver::batchnorm::BnFwdTrainingSpatialMultiple,
                                                 miopen::solver::batchnorm::BnFwdTrainingPerActivation>{};


    const auto slns = solvers.SearchForSolutions(ctx, problem, 1);
    if(slns.empty())
        MIOPEN_THROW(miopenStatusNotImplemented, "No solver found.");

    const auto& sln = slns.front();
    std::cout << sln.solver_id << std::endl;
    if(!sln.invoker_factory)
        MIOPEN_THROW(miopenStatusInternalError, "Invoker missing in solver " + sln.solver_id);

    /*
    for(const auto& sol : solutions){
      auto solver = sol.GetSolver();
                if(solver.IsApplicable(&ctx, &problem))
      std::cout<< sol.ToString()<<std::endl;
    }
    */
    /*
    //TODO: use this call
    //bool IsApplicable(const ExecutionContext& context,
    //                  const miopen::batchnorm::ProblemDescription& problem) const;
    for(const auto& id :
        miopen::solver::GetSolversByPrimitive(miopen::solver::Primitive::Convolution))
    {
        std::cerr << "Testing: " << id.ToString() << std::endl;
        auto solver = id.GetSolver();
        if(id.IsValid() && !solver.IsEmpty())
        {
            try
            {
                if(solver.IsApplicable(ctx))
                    app_solvers.push_back(id.ToString());
            }
            catch(...)
            {
                std::cerr << id.ToString() << "(" << id.Value() << ")"
                          << " raised an exception"
                          << "for " << std::string(network_config) << " config: " << job
                          << std::endl;
            }
        }
        else
        {
            std::cerr << "Solver: " << id.ToString() << " is invalid or empty" << std::endl;
        }
    }*/
    //output["applicable_solvers"] = app_solvers;
    return 0;
}

template <typename Tgpu, typename Tref>
int BNFin<Tgpu, Tref>::GetandSetData()
{

    SetBNDescriptor();
    auto in_len  = GetInputTensorLengths();

    inputTensor = {GetHandle().GetStream(), in_len, is_fwd, is_bwd};

    outputTensor = {GetHandle().GetStream(), in_len, is_fwd, is_bwd};

    if(command["bias"].get<int>() != 0)
    {
        auto bias_len = GetBiasTensorLengths();
        biasScaleTensor    = {GetHandle().GetStream(), bias_len, true, true};
    }

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
    int spatial_dim = command["spatial_dim"];

    if(spatial_dim == 3)
    {
        // NxCxDxHxW -> NxCx(D*H)xW
        return std::vector<int>({in_n, in_c, in_d, in_h, in_w});
    }
    else
    {
        return std::vector<int>({in_n, in_c, in_h, in_w});
    }
}

template <typename Tgpu, typename Tref>
std::vector<int> BNFin<Tgpu, Tref>::GetBiasTensorLengths()
{
    int spatial_dim = command["spatial_dim"];

    std::vector<int> bias_lens(2 + spatial_dim, 1);

    bias_lens[1] = command["out_channels"];

    return bias_lens;
}


template <typename Tgpu, typename Tref>
int BNFin<Tgpu, Tref>::ProcessStep(const std::string& step_name)
{
    steps_processed.push_back(step_name);
    //if(step_name == "alloc_buf")
    //    return AllocateBuffers();
    //if(step_name == "fill_buf")
    //    return FillBuffers();
    //if(step_name == "copy_buf_to_device")
    //    return CopyToDevice();
    //if(step_name == "copy_buf_from_device")
    //    return CopyFromDevice();
    if(step_name == "applicability")
    {
        return TestApplicability();
    }
    //if(step_name == "get_solvers")
    //    return GetSolverList();
    return 0;
}

template <typename Tgpu, typename Tref>
int BNFin<Tgpu, Tref>::SetBNDescriptor()
{
    //    	double bnAlpha = inflags.GetValueDouble("alpha");
    //    	double bnBeta = inflags.GetValueDouble("beta");

    // batch norm mode type
    std::cout << "TEST1" << std::endl;
    if(command["mode"] == 0)
    {
       std::cout << "TEST1.5" << std::endl;
        bn_mode = miopenBNPerActivation;
    }
    else if(command["mode"] == 1)
    {
        std::cout << "TEST2" << std::endl;
        bn_mode = miopenBNSpatial;
    }

    std::cout << "TEST4.6" << std::endl;
    // save off mean and variance?
    if(command["save"] == 0)
    {
    std::cout << "TEST3" << std::endl;
        saveMeanVar = false;
    }
    else if(command["save"] == 1)
    {
    std::cout << "TEST4" << std::endl;
        saveMeanVar = true;
    }

    std::cout << "TEST6.6" << std::endl;
    std::cout << command << std::endl;
    // keep running mean and variance
    if(command["run"] == 0)
    {
    std::cout << "TEST5" << std::endl;
        keepRunningMeanVar = false;
    }
    else if(command["run"] == 1)
    {
    std::cout << "TEST6" << std::endl;
        keepRunningMeanVar = true;
    }

    std::cout << "TEST6" << std::endl;
    forw = command["forw"];
    std::cout << "TEST7" << std::endl;
    back = command["back"];
    std::cout << "TEST8" << std::endl;

    epsilon = 1;
    std::cout << "TEST8" << std::endl;

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
} // namespace fin
#endif // GUARD_MIOPEN_BN_FIN_HPP