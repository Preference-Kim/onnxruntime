// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

#include "core/common/common.h"
#include "core/common/string_join.h"
#include "core/common/safeint.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
namespace webgpu {
class ShaderHelper;
class ComputeContext;
class WebGpuContext;

// data type of uniform variable
enum class ProgramUniformVariableDataType {
  Float32,
  Float16,
  Uint32,
  Int32,
};

constexpr size_t ProgramUniformVariableDataTypeSize[] = {sizeof(float), sizeof(uint16_t), sizeof(uint32_t), sizeof(int32_t)};

constexpr std::string_view ProgramUniformVariableDataTypeName[] = {"f32", "f16", "u32", "i32"};

// represents a runtime value of a uniform variable
struct ProgramUniformVariableValue {
  ProgramUniformVariableValue();  // representing an empty uniform variable
  ProgramUniformVariableValue(float value);
  ProgramUniformVariableValue(uint32_t value);
  ProgramUniformVariableValue(int32_t value);
  ProgramUniformVariableValue(MLFloat16 value);
  ProgramUniformVariableValue(gsl::span<const float> values);
  ProgramUniformVariableValue(gsl::span<const uint32_t> values);
  ProgramUniformVariableValue(gsl::span<const int32_t> values);
  ProgramUniformVariableValue(gsl::span<const MLFloat16> values);

  size_t length;
  ProgramUniformVariableDataType data_type;
  std::vector<uint8_t> data;

 private:
  ProgramUniformVariableValue(ProgramUniformVariableDataType data_type, const void* ptr, size_t element_byte_size, size_t length = 1);
};

// represents a uniform variable definition
struct ProgramUniformVariableDefinition {
  std::string_view name;
  ProgramUniformVariableDataType data_type;
};

// data type of constant
enum class ProgramConstantDataType {
  Float32,
  Float16,
  Uint32,
  Int32,
  Bool
};

constexpr std::string_view ProgramConstantDataTypeName[] = {"f32", "f16", "u32", "i32", "bool"};

// represents a constant in a program
struct ProgramConstant {
  constexpr ProgramConstant(std::string_view name, float value) : name{name}, type{ProgramConstantDataType::Float32}, f32{value} {}
  constexpr ProgramConstant(std::string_view name, uint32_t value) : name{name}, type{ProgramConstantDataType::Uint32}, u32{value} {}
  constexpr ProgramConstant(std::string_view name, int32_t value) : name{name}, type{ProgramConstantDataType::Int32}, i32{value} {}
  constexpr ProgramConstant(std::string_view name, MLFloat16 value) : name{name}, type{ProgramConstantDataType::Float16}, f16{value} {}
  constexpr ProgramConstant(std::string_view name, bool value) : name{name}, type{ProgramConstantDataType::Bool}, boolean{value} {}

  std::string_view name;
  ProgramConstantDataType type;
  union {
    float f32;
    uint32_t u32;
    int32_t i32;
    MLFloat16 f16;
    bool boolean;
  };
};

// represents a runtime value of an overridable constant
struct ProgramOverridableConstantValue {
  constexpr ProgramOverridableConstantValue() : type{}, u32{}, has_value{false} {}  // representing not overriding
  constexpr ProgramOverridableConstantValue(float value) : type{ProgramConstantDataType::Float32}, f32{value}, has_value{true} {}
  constexpr ProgramOverridableConstantValue(uint32_t value) : type{ProgramConstantDataType::Uint32}, u32{value}, has_value{true} {}
  constexpr ProgramOverridableConstantValue(int32_t value) : type{ProgramConstantDataType::Int32}, i32{value}, has_value{true} {}
  constexpr ProgramOverridableConstantValue(MLFloat16 value) : type{ProgramConstantDataType::Float16}, f16{value}, has_value{true} {}
  constexpr ProgramOverridableConstantValue(bool value) : type{ProgramConstantDataType::Bool}, boolean{value}, has_value{true} {}

  ProgramConstantDataType type;
  union {
    float f32;
    uint32_t u32;
    int32_t i32;
    MLFloat16 f16;
    bool boolean;
  };
  bool has_value;
};

// represents an overridable constant definition. may or may not have a default value.
struct ProgramOverridableConstantDefinition {
  constexpr ProgramOverridableConstantDefinition(std::string_view name, ProgramConstantDataType type)
      : name{name}, type{type}, u32{}, has_default_value{false} {}
  constexpr ProgramOverridableConstantDefinition(std::string_view name, float value)
      : name{name}, type{ProgramConstantDataType::Float32}, f32{value}, has_default_value{true} {}
  constexpr ProgramOverridableConstantDefinition(std::string_view name, uint32_t value)
      : name{name}, type{ProgramConstantDataType::Uint32}, u32{value}, has_default_value{true} {}
  constexpr ProgramOverridableConstantDefinition(std::string_view name, int32_t value)
      : name{name}, type{ProgramConstantDataType::Int32}, i32{value}, has_default_value{true} {}
  constexpr ProgramOverridableConstantDefinition(std::string_view name, MLFloat16 value)
      : name{name}, type{ProgramConstantDataType::Float16}, f16{value}, has_default_value{true} {}
  constexpr ProgramOverridableConstantDefinition(std::string_view name, bool value)
      : name{name}, type{ProgramConstantDataType::Bool}, boolean{value}, has_default_value{true} {}

  std::string_view name;
  ProgramConstantDataType type;
  union {
    float f32;
    uint32_t u32;
    int32_t i32;
    MLFloat16 f16;
    bool boolean;
  };
  bool has_default_value;
};

// represents whether the program shader depends on the type, rank, or shape of an input/output tensor
enum class ProgramInputTensorDependency : int {
  None = 0,
  Type = 1,
  Rank = 2,
  Shape = 4,
  TypeAndRank = Type | Rank,
  TypeAndShape = Type | Shape,
};

inline ProgramInputTensorDependency operator|(ProgramInputTensorDependency a, ProgramInputTensorDependency b) {
  return (ProgramInputTensorDependency)((int&)a | (int&)b);
}
inline ProgramInputTensorDependency operator&(ProgramInputTensorDependency a, ProgramInputTensorDependency b) {
  return (ProgramInputTensorDependency)((int&)a & (int&)b);
}
inline ProgramInputTensorDependency& operator|=(ProgramInputTensorDependency& a, ProgramInputTensorDependency b) {
  return (ProgramInputTensorDependency&)((int&)a |= (int&)b);
}
inline ProgramInputTensorDependency& operator&=(ProgramInputTensorDependency& a, ProgramInputTensorDependency b) {
  return (ProgramInputTensorDependency&)((int&)a &= (int&)b);
}

struct ProgramInput {
  const Tensor* tensor;
  ProgramInputTensorDependency dependency;
};

constexpr SafeInt<uint32_t> WORKGROUP_SIZE = 64;

// represents the scope of a variable in a shader program.
//
// this is not a full list of all possible variable scopes in shader programs.
// it only includes what are used in WebGPU EP.
enum class ProgramVariableScope {
  Input = 0,   // storage buffer variable with access mode "read"
  Output = 1,  // storage buffer variable with access mode "read_write"
  Local = 2,   // local variable
};

// data type of variable
//
// this is not a full list of all possible data types in shader programs.
// it only includes what are used in WebGPU EP.
enum class ProgramVariableDataType {
  InvalidType = -1,
  Float32,
  Vec2Float32,
  Vec4Float32,
  Float16,
  Vec2Float16,
  Vec4Float16,
  Int32,
  Vec2Int32,
  Vec4Int32,
  Uint32,
  Vec2Uint32,
  Vec4Uint32,
  Int64,
  Uint64,
  Vec4Bool,
};

inline ProgramVariableDataType ToProgramVariableDataType(int32_t element_type, int component = 1);

namespace detail {
class ProgramWrapper;
}

struct ProgramMetadata;

class ProgramBase {
 public:
  //
  // chain-style methods for setting properties
  //

  // set the cache hint for the program
  template <typename... CacheHintArgs>
  ProgramBase& CacheHint(CacheHintArgs&&... args) {
    cache_hint_ = StringJoin("|", std::forward<CacheHintArgs>(args)...);
  }

  ProgramBase& Inputs(std::initializer_list<ProgramInput> inputs);
  ProgramBase& Outputs(std::initializer_list<Tensor*> outputs);

  ProgramBase& WorkgroupDispatchSize(uint32_t x);
  ProgramBase& WorkgroupDispatchSize(uint32_t x, uint32_t y);
  ProgramBase& WorkgroupDispatchSize(uint32_t x, uint32_t y, uint32_t z);

  ProgramBase& UniformVariables(std::initializer_list<ProgramUniformVariableValue> variables);

  ProgramBase& OverridableConstants(std::initializer_list<ProgramOverridableConstantValue> overridable_constants);

  //
  // shader code generation
  //

  virtual Status GenerateShaderCode(ShaderHelper& sh) const = 0;

  //
  // abstract methods for getting metadata
  //
  // A derived class may contain any of the following static members:
  //
  // \code{.cpp}
  //   // define a list of constant that used in the shader program
  //   static constexpr const ProgramConstant constants[] = { ... };
  //
  //   // define a list of overridable constant that used in the shader program
  //   static constexpr const ProgramOverridableConstantDefinition overridable_constants[] = { ... };
  //
  //   // define a list of uniform variables that used in the shader program
  //   static constexpr const ProgramUniformVariableDefinition uniform_variables[] = { ... };
  // \endcode
  //
  // If those static members exist, the value of them will be used to generate the metadata.
  virtual ProgramMetadata GetMetadata() const = 0;

  //
  // Properties Getters
  //

  inline const std::string& Name() const { return name_; }
  inline const std::string& CacheHint() const { return cache_hint_; }
  inline const std::vector<ProgramInput>& Inputs() const { return inputs_; }
  inline const std::vector<Tensor*>& Outputs() const { return outputs_; }
  inline uint32_t WorkgroupDispatchSizeX() const { return workgroup_dispatch_size_x_; }
  inline uint32_t WorkgroupDispatchSizeY() const { return workgroup_dispatch_size_y_; }
  inline uint32_t WorkgroupDispatchSizeZ() const { return workgroup_dispatch_size_z_; }
  inline const std::vector<ProgramUniformVariableValue>& UniformVariables() const { return variables_; }
  inline const std::vector<ProgramOverridableConstantValue>& OverridableConstants() const { return overridable_constants_; }

 protected:
  virtual ~ProgramBase() = default;

 private:
  // Make the constructor private to prevent direct instantiation or inheritance from this class
  // Use the Program template class as base class to create a new program class
  explicit ProgramBase(const std::string& name);

  std::string name_;
  std::string cache_hint_;
  std::vector<ProgramInput> inputs_;
  std::vector<Tensor*> outputs_;

  uint32_t workgroup_dispatch_size_x_;
  uint32_t workgroup_dispatch_size_y_;
  uint32_t workgroup_dispatch_size_z_;

  std::vector<ProgramUniformVariableValue> variables_;
  std::vector<ProgramOverridableConstantValue> overridable_constants_;

  friend class detail::ProgramWrapper;
};

namespace detail {
// class ProgramWrapper is for accessing private constructor of ProgramBase.
// only ProgramWrapper can access the constructor of ProgramBase because ProgramWrapper is the only friend class of
// ProgramBase. This design is used to prevent direct instantiation or inheritance from ProgramBase.
class ProgramWrapper : public ProgramBase {
 protected:
  template <typename... Args>
  ProgramWrapper(Args&&... args) : ProgramBase{std::forward<Args>(args)...} {}
};

#if defined(ORT_WEBGPU_REGISTER_DERIVED_PROGRAM_CLASS_TYPE_CHECK)
#error "macro ORT_WEBGPU_REGISTER_DERIVED_PROGRAM_CLASS_TYPE_CHECK is already defined"
#endif

#define ORT_WEBGPU_REGISTER_DERIVED_PROGRAM_CLASS_TYPE_CHECK(identifier, element_type)                                                                    \
 private:                                                                                                                                                 \
  template <typename U>                                                                                                                                   \
  static auto test_has_##identifier(int)->decltype(U::identifier, std::true_type{}); /* checks if member exists */                                        \
  template <typename...>                                                                                                                                  \
  static auto test_has_##identifier(...)->std::false_type;                                                                                                \
                                                                                                                                                          \
  template <typename U,                                                                                        /* The following type check uses SFINAE */ \
            typename = std::enable_if_t<                                                                       /* to ensure the specific member:       */ \
                                        std::is_array_v<decltype(U::identifier)> &&                            /*  - is array                          */ \
                                        std::is_const_v<decltype(U::identifier)> &&                            /*  - has "const" modifier              */ \
                                        std::is_convertible_v<decltype(U::identifier), const element_type*> && /*  - can convert to a const pointer    */ \
                                        !std::is_member_pointer_v<decltype(&U::identifier)>>>                  /*  - is static                         */ \
  static auto test_has_##identifier##_with_correct_type(int)->std::true_type;                                                                             \
  template <typename...>                                                                                                                                  \
  static auto test_has_##identifier##_with_correct_type(...)->std::false_type;                                                                            \
                                                                                                                                                          \
 public:                                                                                                                                                  \
  static constexpr bool has_##identifier = decltype(test_has_##identifier<T>(0))::value;                                                                  \
  static constexpr bool has_##identifier##_with_correct_type = decltype(test_has_##identifier##_with_correct_type<T>(0))::value

// the following template class checks whether certain static members exist in the derived class (SFINAE)
template <typename T>
class DerivedProgramClassTypeCheck {
  ORT_WEBGPU_REGISTER_DERIVED_PROGRAM_CLASS_TYPE_CHECK(constants, ProgramConstant);
  ORT_WEBGPU_REGISTER_DERIVED_PROGRAM_CLASS_TYPE_CHECK(overridable_constants, ProgramOverridableConstantDefinition);
  ORT_WEBGPU_REGISTER_DERIVED_PROGRAM_CLASS_TYPE_CHECK(uniform_variables, ProgramUniformVariableDefinition);
};

// compile-time tests for the type check
namespace test {

struct TestClass_Empty {};
struct TestClass_0 {
  int b;
};
struct TestClass_1 {
  int a;
};
struct TestClass_2 {
  const int a;
};
struct TestClass_3 {
  const int a[2];
};
struct TestClass_4 {
  static constexpr int a[] = {0};
};
struct TestClass_5 {
  static int a[];
};
struct TestClass_6 {
  static const int a[];
};

template <typename T>
class TestTypeCheck {
  ORT_WEBGPU_REGISTER_DERIVED_PROGRAM_CLASS_TYPE_CHECK(a, int);
};

static_assert(!TestTypeCheck<TestClass_Empty>::has_a);
static_assert(!TestTypeCheck<TestClass_Empty>::has_a_with_correct_type);
static_assert(!TestTypeCheck<TestClass_0>::has_a);
static_assert(!TestTypeCheck<TestClass_0>::has_a_with_correct_type);
static_assert(TestTypeCheck<TestClass_1>::has_a);
static_assert(!TestTypeCheck<TestClass_1>::has_a_with_correct_type);
static_assert(TestTypeCheck<TestClass_2>::has_a);
static_assert(!TestTypeCheck<TestClass_2>::has_a_with_correct_type);
static_assert(TestTypeCheck<TestClass_3>::has_a);
static_assert(!TestTypeCheck<TestClass_3>::has_a_with_correct_type);
static_assert(TestTypeCheck<TestClass_4>::has_a);
static_assert(TestTypeCheck<TestClass_4>::has_a_with_correct_type);
static_assert(TestTypeCheck<TestClass_5>::has_a);
static_assert(!TestTypeCheck<TestClass_5>::has_a_with_correct_type);
static_assert(TestTypeCheck<TestClass_6>::has_a);
static_assert(TestTypeCheck<TestClass_6>::has_a_with_correct_type);

}  // namespace test

#undef ORT_WEBGPU_REGISTER_DERIVED_PROGRAM_CLASS_TYPE_CHECK

}  // namespace detail

struct ProgramMetadata {
  gsl::span<const ProgramConstant> constants;
  gsl::span<const ProgramOverridableConstantDefinition> overridable_constants;
  gsl::span<const ProgramUniformVariableDefinition> uniform_variables;
};

template <typename T>
class Program : public detail::ProgramWrapper {
 public:
  template <typename... Args>
  Program(Args&&... args) : detail::ProgramWrapper{std::forward<Args>(args)...} {}

  virtual ProgramMetadata GetMetadata() const final {
    ProgramMetadata metadata;
    if constexpr (detail::DerivedProgramClassTypeCheck<T>::has_constants) {
      constexpr const ProgramConstant* ptr = T::constants;
      constexpr size_t len = sizeof(T::constants) / sizeof(ProgramConstant);

      static_assert(detail::DerivedProgramClassTypeCheck<T>::has_constants_with_correct_type &&
                        sizeof(T::constants) % sizeof(ProgramConstant) == 0,
                    "Derived class of \"Program\" has member \"constants\" but its type is incorrect. "
                    "Please use macro WEBGPU_PROGRAM_DEFINE_CONSTANTS() to declare constants.");

      metadata.constants = {ptr, len};
    } else {
      metadata.constants = {};
    }

    if constexpr (detail::DerivedProgramClassTypeCheck<T>::has_overridable_constants) {
      constexpr const ProgramOverridableConstantDefinition* ptr = T::overridable_constants;
      constexpr size_t len = sizeof(T::overridable_constants) / sizeof(ProgramOverridableConstantDefinition);

      static_assert(detail::DerivedProgramClassTypeCheck<T>::has_overridable_constants_with_correct_type &&
                        sizeof(T::overridable_constants) % sizeof(ProgramOverridableConstantDefinition) == 0,
                    "Derived class of \"Program\" has member \"overridable_constants\" but its type is incorrect. "
                    "Please use macro WEBGPU_PROGRAM_DEFINE_OVERRIDABLE_CONSTANTS() to declare overridable constants.");

      metadata.overridable_constants = {ptr, len};
    } else {
      metadata.overridable_constants = {};
    }

    if constexpr (detail::DerivedProgramClassTypeCheck<T>::has_uniform_variables) {
      constexpr const ProgramUniformVariableDefinition* ptr = T::uniform_variables;
      constexpr size_t len = sizeof(T::uniform_variables) / sizeof(ProgramUniformVariableDefinition);

      static_assert(detail::DerivedProgramClassTypeCheck<T>::has_uniform_variables_with_correct_type &&
                        sizeof(T::uniform_variables) % sizeof(ProgramUniformVariableDefinition) == 0,
                    "Derived class of \"Program\" has member \"uniform_variables\" but its type is incorrect. "
                    "Please use macro WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES() to declare uniform variables.");

      metadata.uniform_variables = {ptr, len};
    } else {
      metadata.uniform_variables = {};
    }

    return metadata;
  }
};

#define WEBGPU_PROGRAM_DEFINE_CONSTANTS(...) \
  static constexpr const onnxruntime::webgpu::ProgramConstant constants[] = {__VA_ARGS__}

#define WEBGPU_PROGRAM_DEFINE_OVERRIDABLE_CONSTANTS(...) \
  static constexpr const onnxruntime::webgpu::ProgramOverridableConstantDefinition overridable_constants[] = {__VA_ARGS__}

#define WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(...) \
  static constexpr const onnxruntime::webgpu::ProgramUniformVariableDefinition uniform_variables[] = {__VA_ARGS__}

}  // namespace webgpu
}  // namespace onnxruntime
