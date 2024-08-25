// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/unary_elementwise_ops.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {
Status UnaryElementwiseProgramInfo::GenerateShaderCode(ShaderHelper& sh) const {
  const auto& input = sh.AddVariable(ProgramVariableScope::Input,
                                     "x",
                                     ToProgramVariableDataType(Inputs()[0].tensor->GetElementType(), 4),
                                     1);
  const auto& output = sh.AddVariable(ProgramVariableScope::Output,
                                      "y",
                                      ToProgramVariableDataType(Outputs()[0]->GetElementType(), 4),
                                      1);
  sh.AppendImplementation(additional_impl_);
  sh.MainFunctionBody(sh.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.vec_size"),
                      "let a = ", input.GetByOffset("global_idx"), ";\n",
                      output.SetByOffset("global_idx", expression_));

  return Status::OK();
}

#define WEBGPU_ELEMENTWISE_IMPL(OP_TYPE, ...)                                                \
  class OP_TYPE final : public WebGpuKernel {                                                \
   public:                                                                                   \
    OP_TYPE(const OpKernelInfo& info) : WebGpuKernel{info} {}                                \
                                                                                             \
   protected:                                                                                \
    Status ComputeInternal(ComputeContext& context) const override {                         \
      const auto* input_tensor = context.Input(0);                                           \
      auto* output_tensor = context.Output(0, input_tensor->Shape());                        \
      SafeInt<uint32_t> vec_size = (input_tensor->Shape().Size() + 3) / 4;                   \
      UnaryElementwiseProgramInfo program{#OP_TYPE, __VA_ARGS__};                            \
      program                                                                                \
          .Inputs({{input_tensor, ProgramInputTensorDependency::Type}})                      \
          .Outputs({output_tensor})                                                          \
          .WorkgroupDispatchSize((vec_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)           \
          .UniformVariables({{"vec_size", ProgramUniformVariableDataType::u32, &vec_size}}); \
      return context.RunProgram(program);                                                    \
    }                                                                                        \
  };

#define WEBGPU_ELEMENTWISE_KERNEL(OP_TYPE, VERSION, KERNEL_CLASS, TYPE) \
  ONNX_OPERATOR_KERNEL_EX(                                              \
      OP_TYPE, kOnnxDomain, VERSION, kWebGpuExecutionProvider,          \
      KernelDefBuilder().TypeConstraint("T", TYPE),                     \
      KERNEL_CLASS);

#define WEBGPU_ELEMENTWISE_VERSIONED_KERNEL(OP_TYPE, VERSION_FROM, VERSION_TO, KERNEL_CLASS, TYPE) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                               \
      OP_TYPE, kOnnxDomain, VERSION_FROM, VERSION_TO, kWebGpuExecutionProvider,                    \
      KernelDefBuilder().TypeConstraint("T", TYPE),                                                \
      KERNEL_CLASS);

// WEBGPU_ELEMENTWISE_IMPL(Abs, "abs(a)")
// WEBGPU_ELEMENTWISE_VERSIONED_KERNEL(Abs, 6, 12, Abs, WebGpuSupportedFloatTypes())
// WEBGPU_ELEMENTWISE_KERNEL(Abs, 13, Abs, WebGpuSupportedFloatTypes())

}  // namespace webgpu
}  // namespace onnxruntime
