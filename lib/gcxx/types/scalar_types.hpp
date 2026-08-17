// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_TYPES_SCALAR_TYPES_HPP
#define GCXX_TYPES_SCALAR_TYPES_HPP

#include <complex>
#include <gcxx/internal/prologue.hpp>

// Scalar element-type vocabulary for gcxx. Library code refers to element
// types exclusively through these aliases, so backend-native low-precision
// types can be added later without touching call sites:
//   gcxx::f32_t  -> float
//   gcxx::f64_t  -> double
//   gcxx::cf32_t -> std::complex<float>
//   gcxx::cf64_t -> std::complex<double>
//
// Deferred extension point (do NOT add here): f16_t = __half,
// bf16_t = __nv_bfloat16, cf16_t, cbf16_t. Those require CUDA/HIP headers
// (cuda_fp16.h / cuda_bf16.h / hip/hip_fp16.h) and must not leak into this
// portable header; they belong in a backend-gated companion header when
// half/bfloat16 support lands.
GCXX_NAMESPACE_MAIN_BEGIN()

using f32_t = float;
using f64_t = double;

using cf32_t = std::complex<f32_t>;
using cf64_t = std::complex<f64_t>;

GCXX_NAMESPACE_MAIN_END()

#endif
