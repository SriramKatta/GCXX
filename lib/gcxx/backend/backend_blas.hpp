// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BACKEND_BACKEND_BLAS_HPP_
#define GCXX_BACKEND_BACKEND_BLAS_HPP_

#include <gcxx/backend/backend.hpp>

#if GCXX_CUDA_MODE()
#include <gcxx/backend/cuda_blas_backend.hpp>
#elif GCXX_HIP_MODE()
#include <gcxx/backend/hip_blas_backend.hpp>
#endif

#define GCXX_BLAS_BACKEND(name) APPEND_NAME(BLAS_BACKEND, name)

#if GCXX_CUDA_MODE()
#define GCXX_BLAS_STATUS(name) CUBLAS_STATUS_##name
#elif GCXX_HIP_MODE()
#define GCXX_BLAS_STATUS(name) HIPBLAS_STATUS_##name
#endif

#define GCXX_BLAS_OP(name) \
  GCXX_DIRECT_BACKEND_ALT(CUBLAS_OP_##name, HIPBLAS_OP_##name)

#define GCXX_BLAS_SIDE(name) \
  GCXX_DIRECT_BACKEND_ALT(CUBLAS_SIDE_##name, HIPBLAS_SIDE_##name)

#define GCXX_BLAS_FILL_MODE(name) \
  GCXX_DIRECT_BACKEND_ALT(CUBLAS_FILL_MODE_##name, HIPBLAS_FILL_MODE_##name)

#define GCXX_BLAS_GEMM(name) \
  GCXX_DIRECT_BACKEND_ALT(CUBLAS_GEMM_##name, HIPBLAS_GEMM_##name)

#define GCXX_BLAS_POINTER_MODE(name)                  \
  GCXX_DIRECT_BACKEND_ALT(CUBLAS_POINTER_MODE_##name, \
                          HIPBLAS_POINTER_MODE_##name)

#endif
