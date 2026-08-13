// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_BACKEND_BACKEND_BLAS_HANDLES_HPP_
#define GCXX_RUNTIME_BACKEND_BACKEND_BLAS_HANDLES_HPP_

#include <gcxx/backend/backend_blas.hpp>
#include <gcxx/internal/prologue.hpp>

GCXX_NAMESPACE_MAIN_DRIVER_BEGIN()


using deviceBlasStatus_t = GCXX_BLAS_BACKEND(Status_t);
using deviceBlasHandle_t = GCXX_BLAS_BACKEND(Handle_t);

// Compute-routine enum types (cublasOperation_t / cublasPointerMode_t on CUDA;
// hipblas* on HIP).
using deviceBlasOp_t          = GCXX_BLAS_BACKEND(Operation_t);
using deviceBlasPointerMode_t = GCXX_BLAS_BACKEND(PointerMode_t);

inline constexpr deviceBlasStatus_t deviceBlasStatusSuccess =
  GCXX_BLAS_STATUS(SUCCESS);

// Operation constants shared by the Level-3 wrappers.
inline constexpr deviceBlasOp_t deviceBlasOpN = GCXX_BLAS_OP(N);
inline constexpr deviceBlasOp_t deviceBlasOpT = GCXX_BLAS_OP(T);

// Pointer-mode constants: whether scalar arguments (alpha/beta) are read from
// host or device memory by the BLAS compute routines.
inline constexpr deviceBlasPointerMode_t deviceBlasPointerModeHost =
  GCXX_BLAS_POINTER_MODE(HOST);
inline constexpr deviceBlasPointerMode_t deviceBlasPointerModeDevice =
  GCXX_BLAS_POINTER_MODE(DEVICE);

GCXX_NAMESPACE_MAIN_DRIVER_END()

#endif
