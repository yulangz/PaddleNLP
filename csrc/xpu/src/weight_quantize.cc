// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <core/ctx_manager.h>
#include <core/xft_check.h>
#include <core/xft_event.h>
#include <core/xft_params.h>
#include <paddle/phi/backends/xpu/xpu_context.h>
#include <xft/xdnn_plugin.h>
#include <xft/operation/xft_fc_helper.h>

#include "paddle/extension.h"
#include "paddle/phi/core/enforce.h"
#include "xpu/plugin.h"

namespace xftkernel = baidu::xpu::xftkernel;

std::vector<paddle::Tensor> WeightQuantizeKernel(
    const paddle::Tensor& x,
    const std::string& algo,
    const int arch,
    const int group_size){
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  using XPUType = typename XPUTypeTrait<float16>::Type;
  typedef paddle::float16 data_t;
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  int m = x.shape()[0];
  int n = x.shape()[1];
  auto out = paddle::full({m, n}, -1, paddle::DataType::INT8, x.place());
  auto scale = paddle::full({m}, -1, paddle::DataType::FLOAT32, x.place());
  int ret = baidu::xpu::api::quant2d(
      xpu_ctx->x_context(),
      reinterpret_cast<const XPUType*>(x.data<data_t>()),
      out.data<int8_t>(),
      scale.data<float>(),
      m,
      n);
  return {
      out, scale
  };
}

PD_BUILD_OP(weight_quantize_xpu)
    .Inputs({"x"})
    .Outputs({"out", "scale"})
    .Attrs({"algo: std::string", "arch: int", "group_size: int"})
    .SetKernelFn(PD_KERNEL(WeightQuantizeKernel));
