// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_BACKEND_BACKEND_SPARSE_HPP_
#define GCXX_RUNTIME_BACKEND_BACKEND_SPARSE_HPP_

#include <cstdint>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_handles.hpp>
#include <gcxx/runtime_backend/backend_sparse_handles.hpp>
#include <gcxx/sparse/sparse_error.hpp>

GCXX_NAMESPACE_MAIN_DRIVER_BEGIN()

// ╔════════════════════════════════════════════════════════╗
// ║                  Library handle lifecycle               ║
// ╚════════════════════════════════════════════════════════╝

GCXX_FH auto sparseCreate() -> deviceSparseHandle_t {
  deviceSparseHandle_t handle{};
  GCXX_SAFE_SPARSE_CALL(Create, "Failed to create SPARSE handle", &handle);
  return handle;
}

GCXX_FH auto sparseDestroy(deviceSparseHandle_t handle) -> void {
  GCXX_SAFE_SPARSE_CALL(Destroy, "Failed to destroy SPARSE handle", handle);
}

GCXX_FH auto sparseSetStream(deviceSparseHandle_t handle,
                             deviceStream_t stream) -> void {
  GCXX_SAFE_SPARSE_CALL(SetStream, "Failed to set SPARSE stream", handle,
                        stream);
}

GCXX_FH auto sparseGetStream(deviceSparseHandle_t handle) -> deviceStream_t {
  deviceStream_t stream{};
  GCXX_SAFE_SPARSE_CALL(GetStream, "Failed to get SPARSE stream", handle,
                        &stream);
  return stream;
}

GCXX_FH auto sparseGetVersion(deviceSparseHandle_t handle) -> int {
  int version{};
  GCXX_SAFE_SPARSE_CALL(GetVersion, "Failed to get SPARSE version", handle,
                        &version);
  return version;
}

// ╔════════════════════════════════════════════════════════╗
// ║            Generic API: sparse matrix (SpMat)           ║
// ╚════════════════════════════════════════════════════════╝

GCXX_FH auto sparseCreateCsr(deviceSpMatDescr_t* descr, std::int64_t rows,
                             std::int64_t cols, std::int64_t nnz,
                             void* csrRowOffsets, void* csrColInd,
                             void* csrValues,
                             deviceSparseIndexType_t csrRowOffsetsType,
                             deviceSparseIndexType_t csrColIndType,
                             deviceSparseIndexBase_t indexBase,
                             deviceValueType_t valueType) -> void {
  GCXX_SAFE_SPARSE_CALL(CreateCsr, "Failed to create CSR SpMat descriptor",
                        descr, rows, cols, nnz, csrRowOffsets, csrColInd,
                        csrValues, csrRowOffsetsType, csrColIndType, indexBase,
                        valueType);
}

GCXX_FH auto sparseCreateCoo(deviceSpMatDescr_t* descr, std::int64_t rows,
                             std::int64_t cols, std::int64_t nnz,
                             void* cooRowInd, void* cooColInd, void* cooValues,
                             deviceSparseIndexType_t cooIdxType,
                             deviceSparseIndexBase_t indexBase,
                             deviceValueType_t valueType) -> void {
  GCXX_SAFE_SPARSE_CALL(CreateCoo, "Failed to create COO SpMat descriptor",
                        descr, rows, cols, nnz, cooRowInd, cooColInd, cooValues,
                        cooIdxType, indexBase, valueType);
}

GCXX_FH auto sparseCreateCsc(deviceSpMatDescr_t* descr, std::int64_t rows,
                             std::int64_t cols, std::int64_t nnz,
                             void* cscColOffsets, void* cscRowInd,
                             void* cscValues,
                             deviceSparseIndexType_t cscColOffsetsType,
                             deviceSparseIndexType_t cscRowIndType,
                             deviceSparseIndexBase_t indexBase,
                             deviceValueType_t valueType) -> void {
  GCXX_SAFE_SPARSE_CALL(CreateCsc, "Failed to create CSC SpMat descriptor",
                        descr, rows, cols, nnz, cscColOffsets, cscRowInd,
                        cscValues, cscColOffsetsType, cscRowIndType, indexBase,
                        valueType);
}

GCXX_FH auto sparseDestroySpMat(deviceSpMatDescr_t descr) -> void {
  GCXX_SAFE_SPARSE_CALL(DestroySpMat, "Failed to destroy SpMat descriptor",
                        descr);
}

GCXX_FH auto sparseSpMatGetValues(deviceSpMatDescr_t descr) -> void* {
  void* values{};
  GCXX_SAFE_SPARSE_CALL(SpMatGetValues, "Failed to get SpMat values", descr,
                        &values);
  return values;
}

GCXX_FH auto sparseSpMatSetValues(deviceSpMatDescr_t descr,
                                  void* values) -> void {
  GCXX_SAFE_SPARSE_CALL(SpMatSetValues, "Failed to set SpMat values", descr,
                        values);
}

// SpMatGetFormat / SpMatGetIndexBase are status-returning with an out-pointer.
GCXX_FH auto sparseSpMatGetFormat(deviceSpMatDescr_t descr)
  -> deviceSparseFormat_t {
  deviceSparseFormat_t format{};
  GCXX_SAFE_SPARSE_CALL(SpMatGetFormat, "Failed to get SpMat format", descr,
                        &format);
  return format;
}

GCXX_FH auto sparseSpMatGetIndexBase(deviceSpMatDescr_t descr)
  -> deviceSparseIndexBase_t {
  deviceSparseIndexBase_t indexBase{};
  GCXX_SAFE_SPARSE_CALL(SpMatGetIndexBase, "Failed to get SpMat index base",
                        descr, &indexBase);
  return indexBase;
}

// ╔════════════════════════════════════════════════════════╗
// ║             Generic API: dense matrix (DnMat)           ║
// ╚════════════════════════════════════════════════════════╝

GCXX_FH auto sparseCreateDnMat(deviceDnMatDescr_t* descr, std::int64_t rows,
                               std::int64_t cols, std::int64_t ld, void* values,
                               deviceValueType_t valueType,
                               deviceSparseOrder_t order) -> void {
  GCXX_SAFE_SPARSE_CALL(CreateDnMat, "Failed to create DnMat descriptor", descr,
                        rows, cols, ld, values, valueType, order);
}

GCXX_FH auto sparseDestroyDnMat(deviceDnMatDescr_t descr) -> void {
  GCXX_SAFE_SPARSE_CALL(DestroyDnMat, "Failed to destroy DnMat descriptor",
                        descr);
}

GCXX_FH auto sparseDnMatGetValues(deviceDnMatDescr_t descr) -> void* {
  void* values{};
  GCXX_SAFE_SPARSE_CALL(DnMatGetValues, "Failed to get DnMat values", descr,
                        &values);
  return values;
}

GCXX_FH auto sparseDnMatSetValues(deviceDnMatDescr_t descr,
                                  void* values) -> void {
  GCXX_SAFE_SPARSE_CALL(DnMatSetValues, "Failed to set DnMat values", descr,
                        values);
}

// ╔════════════════════════════════════════════════════════╗
// ║             Generic API: sparse vector (SpVec)          ║
// ╚════════════════════════════════════════════════════════╝

GCXX_FH auto sparseCreateSpVec(deviceSpVecDescr_t* descr, std::int64_t size,
                               std::int64_t nnz, void* indices, void* values,
                               deviceSparseIndexType_t idxType,
                               deviceSparseIndexBase_t indexBase,
                               deviceValueType_t valueType) -> void {
  GCXX_SAFE_SPARSE_CALL(CreateSpVec, "Failed to create SpVec descriptor", descr,
                        size, nnz, indices, values, idxType, indexBase,
                        valueType);
}

GCXX_FH auto sparseDestroySpVec(deviceSpVecDescr_t descr) -> void {
  GCXX_SAFE_SPARSE_CALL(DestroySpVec, "Failed to destroy SpVec descriptor",
                        descr);
}

GCXX_FH auto sparseSpVecGetValues(deviceSpVecDescr_t descr) -> void* {
  void* values{};
  GCXX_SAFE_SPARSE_CALL(SpVecGetValues, "Failed to get SpVec values", descr,
                        &values);
  return values;
}

GCXX_FH auto sparseSpVecSetValues(deviceSpVecDescr_t descr,
                                  void* values) -> void {
  GCXX_SAFE_SPARSE_CALL(SpVecSetValues, "Failed to set SpVec values", descr,
                        values);
}

// ╔════════════════════════════════════════════════════════╗
// ║              Generic API: dense vector (DnVec)          ║
// ╚════════════════════════════════════════════════════════╝

GCXX_FH auto sparseCreateDnVec(deviceDnVecDescr_t* descr, std::int64_t size,
                               void* values,
                               deviceValueType_t valueType) -> void {
  GCXX_SAFE_SPARSE_CALL(CreateDnVec, "Failed to create DnVec descriptor", descr,
                        size, values, valueType);
}

GCXX_FH auto sparseDestroyDnVec(deviceDnVecDescr_t descr) -> void {
  GCXX_SAFE_SPARSE_CALL(DestroyDnVec, "Failed to destroy DnVec descriptor",
                        descr);
}

GCXX_FH auto sparseDnVecGetValues(deviceDnVecDescr_t descr) -> void* {
  void* values{};
  GCXX_SAFE_SPARSE_CALL(DnVecGetValues, "Failed to get DnVec values", descr,
                        &values);
  return values;
}

GCXX_FH auto sparseDnVecSetValues(deviceDnVecDescr_t descr,
                                  void* values) -> void {
  GCXX_SAFE_SPARSE_CALL(DnVecSetValues, "Failed to set DnVec values", descr,
                        values);
}

GCXX_NAMESPACE_MAIN_DRIVER_END()

#endif
