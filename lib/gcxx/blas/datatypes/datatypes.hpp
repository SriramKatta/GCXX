// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_DATATYPES_DATATYPES_HPP_
#define GCXX_BLAS_DATATYPES_DATATYPES_HPP_

#include <complex>
#include <cstdint>

#include <gcxx/backend/backend_blas.hpp>
#include <gcxx/internal/prologue.hpp>
#include <gcxx/types/scalar_types.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

// Map cpp scalar type to its backend data-type enum
template <typename VT>
struct cuda_datatype {
  static_assert(gcxx::details_::is_always_false_v<VT>,
                "unsupported blas datatype");
};

#define DEFINE_DATATYPE(CPP_TYPE, CUDA_ENUM, HIP_ENUM) \
  template <>                                          \
  struct cuda_datatype<CPP_TYPE> {                     \
    static constexpr auto datatype =                   \
      GCXX_DIRECT_BACKEND_ALT(CUDA_ENUM, HIP_ENUM);    \
  }

// ╔════════════════════════════════════════════════════════╗
// ║                  real floating point                   ║
// ╚════════════════════════════════════════════════════════╝
DEFINE_DATATYPE(gcxx::f32_t, CUDA_R_32F, HIP_R_32F);
DEFINE_DATATYPE(gcxx::f64_t, CUDA_R_64F, HIP_R_64F);

// ╔════════════════════════════════════════════════════════╗
// ║                 complex floating point                 ║
// ╚════════════════════════════════════════════════════════╝
DEFINE_DATATYPE(gcxx::cf32_t, CUDA_C_32F, HIP_C_32F);
DEFINE_DATATYPE(gcxx::cf64_t, CUDA_C_64F, HIP_C_64F);

// ╔════════════════════════════════════════════════════════╗
// ║                        integer                         ║
// ╚════════════════════════════════════════════════════════╝
DEFINE_DATATYPE(gcxx::i8_t, CUDA_R_8I, HIP_R_8I);
DEFINE_DATATYPE(gcxx::u8_t, CUDA_R_8U, HIP_R_8U);
DEFINE_DATATYPE(gcxx::i32_t, CUDA_R_32I, HIP_R_32I);

#undef DEFINE_DATATYPE

template <typename VT>
inline constexpr auto cuda_datatype_v = cuda_datatype<VT>::datatype;

// Map a cpp scalar type to its backend compute-type enum
template <typename VT>
struct blas_compute_type {
  static_assert(gcxx::details_::is_always_false_v<VT>,
                "unsupported blas compute type");
};

#define DEFINE_COMPUTE_TYPE(CPP_TYPE, CUDA_ENUM, HIP_ENUM) \
  template <>                                              \
  struct blas_compute_type<CPP_TYPE> {                     \
    static constexpr auto compute_type =                   \
      GCXX_DIRECT_BACKEND_ALT(CUDA_ENUM, HIP_ENUM);        \
  }

// ╔════════════════════════════════════════════════════════╗
// ║                  real floating point                   ║
// ╚════════════════════════════════════════════════════════╝
DEFINE_COMPUTE_TYPE(gcxx::f32_t, CUBLAS_COMPUTE_32F, HIPBLAS_COMPUTE_32F);
DEFINE_COMPUTE_TYPE(gcxx::f64_t, CUBLAS_COMPUTE_64F, HIPBLAS_COMPUTE_64F);

#undef DEFINE_COMPUTE_TYPE

template <typename VT>
inline constexpr auto blas_compute_type_v = blas_compute_type<VT>::compute_type;

GCXX_NAMESPACE_MAIN_BLAS_END()


#endif
