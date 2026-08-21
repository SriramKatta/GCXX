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

using deviceBlasOp_t          = GCXX_BLAS_BACKEND(Operation_t);
using deviceBlasPointerMode_t = GCXX_BLAS_BACKEND(PointerMode_t);
using deviceBlasSideMode_t    = GCXX_BLAS_BACKEND(SideMode_t);
using deviceBlasFillMode_t    = GCXX_BLAS_BACKEND(FillMode_t);
using deviceBlasDiagType_t    = GCXX_BLAS_BACKEND(DiagType_t);

inline constexpr deviceBlasStatus_t deviceBlasStatusSuccess =
  GCXX_BLAS_STATUS(SUCCESS);

// Operation constants shared by the Level-3 wrappers.
inline constexpr deviceBlasOp_t deviceBlasOpN = GCXX_BLAS_OP(N);
inline constexpr deviceBlasOp_t deviceBlasOpT = GCXX_BLAS_OP(T);

// Side mode: which side of the matrix the vector operand applies to (dgmm).
inline constexpr deviceBlasSideMode_t deviceBlasSideLeft = GCXX_BLAS_SIDE(LEFT);
inline constexpr deviceBlasSideMode_t deviceBlasSideRight =
  GCXX_BLAS_SIDE(RIGHT);

// Fill mode: which triangle of a symmetric/hermitian operand is stored.
inline constexpr deviceBlasFillMode_t deviceBlasFillModeUpper =
  GCXX_BLAS_FILL_MODE(UPPER);
inline constexpr deviceBlasFillMode_t deviceBlasFillModeLower =
  GCXX_BLAS_FILL_MODE(LOWER);

// Diag type: unit diagonal assumed vs stored diagonal (trsm/trsv family).
inline constexpr deviceBlasDiagType_t deviceBlasDiagNonUnit =
  GCXX_BLAS_DIAG(NON_UNIT);
inline constexpr deviceBlasDiagType_t deviceBlasDiagUnit = GCXX_BLAS_DIAG(UNIT);

// Pointer mode: whether alpha/beta scalars are host- or device-resident.
inline constexpr deviceBlasPointerMode_t deviceBlasPointerModeHost =
  GCXX_BLAS_POINTER_MODE(HOST);
inline constexpr deviceBlasPointerMode_t deviceBlasPointerModeDevice =
  GCXX_BLAS_POINTER_MODE(DEVICE);

GCXX_NAMESPACE_MAIN_DRIVER_END()

#endif
