// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_MACROS_FUNCTION_DECORATOR_MACROS_HPP_
#define GCXX_MACROS_FUNCTION_DECORATOR_MACROS_HPP_

#ifndef GCXX_H
#define GCXX_H() __host__
#endif

#ifndef GCXX_FH
#define GCXX_FH() __forceinline__ __host__
#endif

#ifndef GCXX_FC
#define GCXX_FC() __forceinline__ constexpr
#endif

#ifndef GCXX_FHC
#define GCXX_FHC() __forceinline__ __host__ constexpr
#endif

#ifndef GCXX_D
#define GCXX_D() __device__
#endif

#ifndef GCXX_FD
#define GCXX_FD() __forceinline__ __device__
#endif

#ifndef GCXX_FDC
#define GCXX_FDC() __forceinline__ __device__ constexpr
#endif

#ifndef GCXX_HD
#define GCXX_HD() __host__ __device__
#endif

#ifndef GCXX_FHD
#define GCXX_FHD() __forceinline__ __host__ __device__
#endif

#ifndef GCXX_FHDC
#define GCXX_FHDC() __forceinline__ __host__ __device__ constexpr
#endif

#ifndef GCXX_CXPR
#define GCXX_CXPR() constexpr
#endif

#ifndef GCXX_NOEXCEPT
#define GCXX_NOEXCEPT() noexcept
#endif


#ifndef GCXX_CONST_NOEXCEPT
#define GCXX_CONST_NOEXCEPT() const noexcept
#endif

#if GCXX_CUDA_MODE
#define GCXXRT_CB() CUDART_CB
#else
#define GCXXRT_CB()
#endif

#if defined(__INTEL_COMPILER)
#define GCXX_RESTRICT_KEYWORD() __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define GCXX_RESTRICT_KEYWORD() __restrict__
#else
#define GCXX_RESTRICT_KEYWORD()
#endif

#endif
