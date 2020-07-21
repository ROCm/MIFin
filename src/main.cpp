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
#include <miopen/miopen.h>

#include "fin.hpp"
#include "conv_fin.hpp"
#include "error.hpp"

#include <miopen/tensor.hpp>
#include <nlohmann/json.hpp>


#include <half.hpp>
#include <algorithm>
#include <cstdio>
#include <iostream>

using json = nlohmann::json;

// TODO: implement in the tensor class
/*
std::vector<int> GetTensorLengths(const miopen::TensorDescriptor &tensor) {

  std::vector<int> tensor_len;
  tensor_len.resize(tensor.GetSize());
  std::copy(tensor.GetLengths().begin(), tensor.GetLengths().end(),
            tensor_len.data());

  return tensor_len;
}

// TODO: implement int he tensor class
size_t GetTensorSize(const miopen::TensorDescriptor &tensor) {
  std::vector<int> len = GetTensorLengths(tensor);
  size_t sz =
      std::accumulate(len.begin(), len.end(), 1, std::multiplies<int>());

  return sz;
}
*/
int main(int argc, char *argv[]) 
{
    if(argc != 3)
    {
        std::cerr << "Fin requires two arguments" << std::endl;
    }
    boost::filesystem::path input_filename(argv[1]);
    if(!boost::filesystem::exists(input_filename))
    {
        std::cerr << "File: " << input_filename.string() << " does not exist" << std::endl;
        exit(-1);
    }

    boost::filesystem::path output_filename(argv[2]);


    // The JSON is a list of commands, so we iterate over the list and then process each map
    std::ifstream i(input_filename.string());
    // TODO: fix the output writing so that interim results are not lost if one of the iterations crash
    std::ofstream o(output_filename.string());
    json j; //  = json::parse(cmd);
    i >> j;
    i.close();
    json final_output;
    for(auto& it : j)
    {
        auto command = it;
        std::cout << it << std::endl;
        fin::Fin* f = nullptr;
        // TODO : Move this to a factory function
        if(command["config"]["cmd"] == "conv")
        {
            f = new fin::ConvFin<float, float>(command);
        }
        else
        {
            FIN_THROW("All hell breaks loose");
            exit(-1);
        }

        for(auto & step_it : command["steps"])
        {
            std::string step = step_it.get<std::string>();
            std::cout << "Processing step: " << step << std::endl;
            f->ProcessStep(step);           
        }
        final_output.push_back(f->output);
    }
    o << std::setw(4) << final_output << std::endl;
  return 0;
}

// used for dev/debug
    /*
    const std::string cmd = R"([{ "steps": ["alloc_buf", "fill_buf", "copy_buf_to_device", "copy_buf_from_device", "applicability"], "tag" : "resnet50", "label" : "resnet_tuning", "direction" : 4, "arch" : "gfx906", "num_cu" : 64, "config" : { "in_w" : 28, "sources" : [ "issue_1760" ], "pad_d" : 0, "out_channels" : 128, "dilation_d" : 1, "pad_w" : 1, "conv_stride_h" : 1, "conv_stride_d" : 1, "fusion_mode" : -1, "pad_mode" : "default", "in_h" : 28, "tags" : [ "resnet50" ], "in_d" : 1, "cmd" : "conv", "activMode" : -1, "fil_h" : 3, "group_count" : 1, "dilation_h" : 1, "in_channels" : 128, "pad_h" : 1, "batchsize" : 32, "conv_stride_w" : 1, "conv_mode" : "conv", "recur" : 0, "fil_w" : 3, "spatial_dim" : 2, "fil_d" : 1, "trans_output_pad_d" : 0, "dilation_w" : 1 } } ])";
*/
