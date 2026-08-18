// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_TYPES_SCALAR_TYPES_HPP
#define GCXX_TYPES_SCALAR_TYPES_HPP

#include <complex>
#include <cstdint>
#include <gcxx/internal/prologue.hpp>

// TODO : Deferred extension point
// using f16_t = __half,
// using  bf16_t = __nv_bfloat16 etc etc
// These require CUDA/HIP headers (cuda_fp16.h / cuda_bf16.h / hip/hip_fp16.h)
// and must not leak into this portable header; they belong in a backend-gated
// companion header when half/bfloat16 support lands.
GCXX_NAMESPACE_MAIN_BEGIN()

using f32_t = float;
using f64_t = double;

using cf32_t = std::complex<f32_t>;
using cf64_t = std::complex<f64_t>;

using i8_t  = std::int8_t;
using u8_t  = std::uint8_t;
using i32_t = std::int32_t;

GCXX_NAMESPACE_MAIN_END()

#endif
