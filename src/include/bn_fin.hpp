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
#include <miopen/convolution.hpp>
#include <miopen/find_db.hpp>
#include <miopen/invoker.hpp>
#include <miopen/load_file.hpp>
#include <miopen/md5.hpp>
#include <miopen/perf_field.hpp>
#include <miopen/solver_id.hpp>

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

    tensor<Tgpu, Tcpu> inputTensor;
    tensor<Tgpu, Tcpu> inputTensor_vect4;
    tensor<Tgpu, Tcpu> outputTensor;
    //tensor<Tgpu, Tcpu> biasTensor;
    //tensor<Tgpu, Tcpu> workspace;
    miopen::ConvolutionDescriptor convDesc;

    bool wrw_allowed = 0, bwd_allowed = 0, forward_allowed = 1;
    bool is_fwd            = true;
    bool is_bwd            = false;
    bool is_wrw            = false; // TODO: check redundancy with above
    int immediate_solution = 0;
    miopenBatchNormMode_t bn_mode;
    bool saveMeanVar;
    bool keepRunningMeanVar;
    std::vector<std::string> steps_processed;
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
    const auto problem = miopen::ProblemDescription{bn_mode,
                                                       xDesc,
                                                       yDesc,
                                                       bnScaleBiasMeanVarDesc,
                                                       expAvgFactor,
                                                       epsilon,
                                                       resultsave,
                                                       resultrunning};

    auto ctx    = miopen::ConvolutionContext{problem};
    auto handle = miopen::Handle{};
#if MIOPEN_MODE_NOGPU
    InitNoGpuHandle(handle);
#else
    throw std::runtime_error("MIOpen needs to be compiled with the NOGPU backend "
                             "to test applicability");
#endif

    ctx.SetStream(&handle);
    ctx.DetectRocm();
    ctx.SetupFloats();
    const auto network_config = ctx.BuildConfKey();
    std::vector<std::string> app_solvers;
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
    }
    output["applicable_solvers"] = app_solvers;
    return 0;
}

template <typename Tgpu, typename Tref>
int BNFin<Tgpu, Tref>::GetandSetData()
{

    SetBNDescriptor()
    auto in_len  = GetInputTensorLengths();

    // auto y_type = GetOutputType();

    inputTensor = {GetHandle().GetStream(), in_len, is_bwd};

    // conv, input and weight tensor descriptors need to be set before we can know the
    // output lengths
    auto out_len = GetOutputTensorLengths();
    outputTensor = {GetHandle().GetStream(), out_len, (is_bwd || is_wrw), is_fwd};

    if(IsInputTensorTransform())
    {
        std::vector<int> in_len_v4(in_len.begin(), in_len.end());
        in_len_v4[1] = ((in_len[1] + 3) / 4) * 4;

        inputTensor_vect4  = {GetHandle().GetStream(), in_len_v4, (is_fwd || is_wrw), is_bwd};
    }


    return (0);
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

    return miopenStatusSuccess;
}
} // namespace fin
#endif // GUARD_MIOPEN_BN_FIN_HPP
