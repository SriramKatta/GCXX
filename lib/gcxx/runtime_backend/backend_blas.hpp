// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_BACKEND_BACKEND_BLAS_HPP_
#define GCXX_RUNTIME_BACKEND_BACKEND_BLAS_HPP_

#include <gcxx/internal/prologue.hpp>

#include <gcxx/blas/blas_error.hpp>
#include <gcxx/runtime_backend/backend_blas_handles.hpp>
#include <gcxx/runtime_backend/backend_handles.hpp>

GCXX_NAMESPACE_MAIN_DRIVER_BEGIN()

GCXX_FH auto blasCreate() -> deviceBlasHandle_t {
  deviceBlasHandle_t handle{};
  GCXX_SAFE_BLAS_CALL(Create, "Failed to create BLAS handle", &handle);
  return handle;
}

GCXX_FH auto blasDestroy(deviceBlasHandle_t handle) -> void {
  GCXX_SAFE_BLAS_CALL(Destroy, "Failed to destroy BLAS handle", handle);
}

GCXX_FH auto blasSetStream(deviceBlasHandle_t handle,
                           deviceStream_t stream) -> void {
  GCXX_SAFE_BLAS_CALL(SetStream, "Failed to set BLAS stream", handle, stream);
}

GCXX_FH auto blasGetStream(deviceBlasHandle_t handle) -> deviceStream_t {
  deviceStream_t stream{};
  GCXX_SAFE_BLAS_CALL(GetStream, "Failed to get BLAS stream", handle, &stream);
  return stream;
}

GCXX_FH auto blasGetVersion(deviceBlasHandle_t handle) -> int {
  int version{};
  GCXX_SAFE_BLAS_CALL(GetVersion, "Failed to get BLAS version", handle,
                      &version);
  return version;
}

GCXX_NAMESPACE_MAIN_DRIVER_END()

#endif
