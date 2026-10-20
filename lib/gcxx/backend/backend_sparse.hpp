// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BACKEND_BACKEND_SPARSE_HPP_
#define GCXX_BACKEND_BACKEND_SPARSE_HPP_

#include <gcxx/backend/backend.hpp>

#if GCXX_CUDA_MODE()
#include <gcxx/backend/cuda_sparse_backend.hpp>
#elif GCXX_HIP_MODE()
#include <gcxx/backend/hip_sparse_backend.hpp>
#endif

#define GCXX_SPARSE_BACKEND(name) APPEND_NAME(SPARSE_BACKEND, name)

#if GCXX_CUDA_MODE()
#define GCXX_SPARSE_STATUS(name) CUSPARSE_STATUS_##name
#elif GCXX_HIP_MODE()
#define GCXX_SPARSE_STATUS(name) HIPSPARSE_STATUS_##name
#endif

#endif
