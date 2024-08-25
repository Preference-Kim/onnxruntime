// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {

class UnaryElementwiseProgramInfo final : public Program<UnaryElementwiseProgramInfo> {
 public:
  UnaryElementwiseProgramInfo(const std::string& kernel_name, const std::string& expression, const std::string& additional_impl = "")
      : Program{kernel_name}, expression_{expression}, additional_impl_{additional_impl} {
  }

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_CONSTANTS(
      {"x", 3.14f},
      {"y", 0});

  static constexpr const ProgramUniformVariableDefinition uniform_variable_definitions[] = {
      {"input", ProgramUniformVariableDataType::Float32},
      {"output", ProgramUniformVariableDataType::Float32},
      {"vec_size", ProgramUniformVariableDataType::Uint32},
  };

 private:
  std::string expression_;
  std::string additional_impl_;
};

}  // namespace webgpu
}  // namespace onnxruntime
