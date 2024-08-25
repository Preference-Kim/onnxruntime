// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/program_cache_key.h"

namespace onnxruntime {
namespace webgpu {

std::string CalculateProgramCacheKey(const ProgramBase& program, bool is_1d_dispatch) {
  std::ostringstream ss;
  ss.imbue(std::locale::classic());

  // final key format:
  // <KEY>=<PROGRAM_NAME>[<PROGRAM_CUSTOM_CACHE_HINT>]:is1DimensionDispatch:<UNIFORMS>:<INPUTS_INFO>
  //
  // <PROGRAM_CUSTOM_CACHE_HINT>=<HINT_0>|<HINT_1>|...
  // <UNIFORMS>=<UNIFORMS_INFO_0>|<UNIFORMS_INFO_1>|...
  // <UNIFORMS_INFO_i>=<UNIFORM_LENGTH>
  // <INPUTS_INFO>=<INPUTS_INFO_0>|<INPUTS_INFO_1>|...
  // <INPUTS_INFO_i>=<TENSOR_ELEMENT_TYPE_OR_EMPTY>;<TENSOR_SHAPE_OR_RANK_OR_EMPTY>
  ss << program.Name();
  auto& hint = program.CacheHint();
  if (!hint.empty()) {
    ss << "[" << program.CacheHint() << "]";
  }
  ss << ":" << is_1d_dispatch << ":";
  bool first = true;
  for (const auto& uniform : program.UniformVariables()) {
    if (first) {
      first = false;
    } else {
      ss << "|";
    }
    if (uniform.length > 0) {
      ss << uniform.length;
    }
  }
  ss << ":";
  first = true;
  for (const auto& input : program.Inputs()) {
    if (first) {
      first = false;
    } else {
      ss << "|";
    }
    if ((input.dependency & ProgramInputTensorDependency::Type) == ProgramInputTensorDependency::Type) {
      ss << input.tensor->GetElementType();
    }
    ss << ";";
    if ((input.dependency & ProgramInputTensorDependency::Rank) == ProgramInputTensorDependency::Rank) {
      ss << input.tensor->Shape().NumDimensions();
    } else if ((input.dependency & ProgramInputTensorDependency::Shape) == ProgramInputTensorDependency::Shape) {
      ss << input.tensor->Shape().ToString();
    }
  }

  return ss.str();
}

}  // namespace webgpu
}  // namespace onnxruntime
