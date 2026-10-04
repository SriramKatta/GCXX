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

GCXX_FH auto blasSetPointerMode(deviceBlasHandle_t handle,
                                deviceBlasPointerMode_t mode) -> void {
  GCXX_SAFE_BLAS_CALL(SetPointerMode, "Failed to set BLAS pointer mode", handle,
                      mode);
}

GCXX_FH auto blasGetPointerMode(deviceBlasHandle_t handle)
  -> deviceBlasPointerMode_t {
  deviceBlasPointerMode_t mode{};
  GCXX_SAFE_BLAS_CALL(GetPointerMode, "Failed to get BLAS pointer mode", handle,
                      &mode);
  return mode;
}

GCXX_FH auto blasGetVersion(deviceBlasHandle_t handle) -> int {
#if GCXX_CUDA_MODE()
  int version{};
  GCXX_SAFE_BLAS_CALL(GetVersion, "Failed to get BLAS version", handle,
                      &version);
  return version;
#elif GCXX_HIP_MODE()
  // hipBLAS exposes the version only at compile time
  (void)handle;
  constexpr int version_scale_major = 10000;
  constexpr int version_scale_minor = 100;
  return hipblasVersionMajor * version_scale_major +
         hipblasVersionMinor * version_scale_minor + hipblasVersionPatch;
#endif
}

GCXX_FH auto blasSetVector(int n, int elemsize, const void* hostptr,
                           int hostinc, void* devptr, int devinc) {
  GCXX_SAFE_BLAS_CALL(SetVector, "Failed to set vector", n, elemsize, hostptr,
                      hostinc, devptr, devinc);
}

GCXX_FH auto blasSetVectorAsync(deviceStream_t str, int n, int elemsize,
                                const void* hostptr, int hostinc, void* devptr,
                                int devinc) {
  GCXX_SAFE_BLAS_CALL(SetVectorAsync, "Failed to set vector", n, elemsize,
                      hostptr, hostinc, devptr, devinc, str);
}

GCXX_FH auto blasGetVector(int n, int elemsize, const void* devptr, int devinc,
                           void* hostptr, int hostinc) {
  GCXX_SAFE_BLAS_CALL(GetVector, "Failed to get vector", n, elemsize, devptr,
                      devinc, hostptr, hostinc);
}

GCXX_FH auto blasGetVectorAsync(deviceStream_t str, int n, int elemsize,
                                const void* devptr, int devinc, void* hostptr,
                                int hostinc) {
  GCXX_SAFE_BLAS_CALL(GetVectorAsync, "Failed to get vector", n, elemsize,
                      devptr, devinc, hostptr, hostinc, str);
}

GCXX_FH auto blasSetMatrix(int rows, int cols, int elemsize,
                           const void* hostptr, int hostld, void* devptr,
                           int devld) {
  GCXX_SAFE_BLAS_CALL(SetMatrix, "Failed to set matrix", rows, cols, elemsize,
                      hostptr, hostld, devptr, devld);
}

GCXX_FH auto blasSetMatrixAsync(deviceStream_t str, int rows, int cols,
                                int elemsize, const void* hostptr, int hostld,
                                void* devptr, int devld) {
  GCXX_SAFE_BLAS_CALL(SetMatrixAsync, "Failed to set matrix", rows, cols,
                      elemsize, hostptr, hostld, devptr, devld, str);
}

GCXX_FH auto blasGetMatrix(int rows, int cols, int elemsize, const void* devptr,
                           int devld, void* hostptr, int hostld) {
  GCXX_SAFE_BLAS_CALL(GetMatrix, "Failed to get matrix", rows, cols, elemsize,
                      devptr, devld, hostptr, hostld);
}

GCXX_FH auto blasGetMatrixAsync(deviceStream_t str, int rows, int cols,
                                int elemsize, const void* devptr, int devld,
                                void* hostptr, int hostld) {
  GCXX_SAFE_BLAS_CALL(GetMatrixAsync, "Failed to get matrix", rows, cols,
                      elemsize, devptr, devld, hostptr, hostld, str);
}

GCXX_NAMESPACE_MAIN_DRIVER_END()

#endif
