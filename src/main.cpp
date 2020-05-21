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

#include <miopen/tensor.hpp>

#include <half.hpp>

#include <algorithm>
#include <cstdio>
#include <iostream>

// TODO: implement in the tensor class
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

int main(int argc, char *argv[]) {
  // show command
  std::cout << "fin:";
  for (int i = 1; i < argc; i++)
    std::cout << " " << argv[i];
  std::cout << std::endl;

    fin::Fin* f = nullptr;
    f = new fin::ConvFin<float, float>();
    (void)f;
  // auto f = std::make_unique<fin::ConvFin>(new fin::ConvFin<float, float>());
#if 0
  else if(base_arg == "convfp16")
  {
      tuner = std::make_unique<ConvTuner<float16, float>>();
  }
  else if(base_arg == "convbfp16")
  {
      tuner = std::make_unique<ConvTuner<bfloat16, float>>();
  }
  else
  {
      std::cout << "Incorrect BaseArg\n";
      exit(0);
  }
#endif
#if 0
  int fargval = tuner->GetInputFlags().GetValueInt("forw");

  // int iter = tuner->GetInputFlags().GetValueInt("iter");
  int status;

  if(fargval & 1 || fargval == 0)
  {
      status = tuner->RunForwardGPU();
  }

  if(fargval != 1)
  {
      status = tuner->RunBackwardGPU();
  }
#endif
  return 0;
}
