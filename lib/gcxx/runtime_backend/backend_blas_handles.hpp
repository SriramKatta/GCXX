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

inline constexpr deviceBlasStatus_t deviceBlasStatusSuccess =
  GCXX_BLAS_STATUS(SUCCESS);

GCXX_NAMESPACE_MAIN_DRIVER_END()

#endif
