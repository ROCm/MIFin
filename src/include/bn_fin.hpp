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
    BNFin(json _job) : Fin(), job(_job){}

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
        // timing is always enabled
        is_fwd = (job["direction"].get<int>() == 0 || job["direction"].get<int>() & 1);
        is_bwd = (job["direction"].get<int>() == 0 || job["direction"].get<int>() & 2);
        is_wrw = (job["direction"].get<int>() == 0 || job["direction"].get<int>() & 4);
        SetBNDescriptor();
        // workspace_dev = nullptr; // TODO: replaced with a tensor class
        // the variable name is implementation dependent, checking size instead
    }


    // Getters and setters
    std::vector<int> GetInputTensorLengths();
    std::vector<int> GetBiasTensorLengths();
    int SetBNDescriptor();
    std::vector<size_t> GetOutputTensorLengths() const;
    miopenDataType_t GetOutputType() const
    {
        return (data_type == miopenInt8 || data_type == miopenInt8x4) ? miopenFloat : data_type;
    }
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

    int forw = 0;
    int back = 1;
    bool is_fwd            = true;
    bool is_bwd            = false;
    bool is_wrw            = false; // TODO: check redundancy with above

    const miopen::TensorDescriptor inputTensor;
    const miopen::TensorDescriptor biasScaleTensor;
    const miopen::TensorDescriptor outputTensor;
    //miopenTensorDescriptor_t biasScaleTensor;
    //miopenTensorDescriptor_t outputTensor;
    //tensor<Tgpu, Tcpu> inputTensor;
    //tensor<Tgpu, Tcpu> outputTensor;
    //tensor<Tgpu, Tcpu> biasScaleTensor;

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

    //SetTensorNd(inputTensor, in_len, data_type);
    //SetTensorNd(biasScaleTensor, sb_len, ((sizeof(Tgpu) == 4) ? miopenFloat : miopenHalf));
    //SetTensorNd(outputTensor, in_len, data_type);

    // backwards
    //SetTensorNd(dyInputTensor, in_len, data_type);
    //SetTensorNd(dxOutputTensor, in_len, data_type);


    return (0);
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

    epsilon = command["epsilon"];

    return miopenStatusSuccess;
}
} // namespace fin
#endif // GUARD_MIOPEN_BN_FIN_HPP
