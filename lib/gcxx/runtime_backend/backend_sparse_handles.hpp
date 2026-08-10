// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_BACKEND_BACKEND_SPARSE_HANDLES_HPP_
#define GCXX_RUNTIME_BACKEND_BACKEND_SPARSE_HANDLES_HPP_

#include <gcxx/backend/backend_sparse.hpp>
#include <gcxx/internal/prologue.hpp>

GCXX_NAMESPACE_MAIN_DRIVER_BEGIN()

// ── Library status / handle ────────────────────────────────────────────────
using deviceSparseStatus_t = GCXX_SPARSE_BACKEND(Status_t);
using deviceSparseHandle_t = GCXX_SPARSE_BACKEND(Handle_t);

// ── Generic API opaque descriptors ─────────────────────────────────────────
using deviceSpMatDescr_t = GCXX_SPARSE_BACKEND(SpMatDescr_t);
using deviceDnMatDescr_t = GCXX_SPARSE_BACKEND(DnMatDescr_t);
using deviceSpVecDescr_t = GCXX_SPARSE_BACKEND(SpVecDescr_t);
using deviceDnVecDescr_t = GCXX_SPARSE_BACKEND(DnVecDescr_t);

// ── Generic API enums (cuSPARSE/hipSPARSE names align) ─────────────────────
using deviceSparseIndexType_t = GCXX_SPARSE_BACKEND(IndexType_t);
using deviceSparseIndexBase_t = GCXX_SPARSE_BACKEND(IndexBase_t);
using deviceSparseFormat_t    = GCXX_SPARSE_BACKEND(Format_t);
using deviceSparseOrder_t     = GCXX_SPARSE_BACKEND(Order_t);

// Value type diverges between backends: cudaDataType_t vs hipDataType.
using deviceValueType_t = GCXX_DIRECT_BACKEND_ALT(cudaDataType_t, hipDataType);

inline constexpr deviceSparseStatus_t deviceSparseStatusSuccess =
  GCXX_SPARSE_STATUS(SUCCESS);

// ── Common enum constants (enumerator names diverge, alias directly) ───────
inline constexpr deviceSparseIndexType_t deviceSparseIndex16U =
  GCXX_DIRECT_BACKEND_ALT(CUSPARSE_INDEX_16U, HIPSPARSE_INDEX_16U);
inline constexpr deviceSparseIndexType_t deviceSparseIndex32I =
  GCXX_DIRECT_BACKEND_ALT(CUSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I);
inline constexpr deviceSparseIndexType_t deviceSparseIndex64I =
  GCXX_DIRECT_BACKEND_ALT(CUSPARSE_INDEX_64I, HIPSPARSE_INDEX_64I);

inline constexpr deviceSparseIndexBase_t deviceSparseIndexBaseZero =
  GCXX_DIRECT_BACKEND_ALT(CUSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ZERO);
inline constexpr deviceSparseIndexBase_t deviceSparseIndexBaseOne =
  GCXX_DIRECT_BACKEND_ALT(CUSPARSE_INDEX_BASE_ONE, HIPSPARSE_INDEX_BASE_ONE);

inline constexpr deviceSparseFormat_t deviceSparseFormatCsr =
  GCXX_DIRECT_BACKEND_ALT(CUSPARSE_FORMAT_CSR, HIPSPARSE_FORMAT_CSR);
inline constexpr deviceSparseFormat_t deviceSparseFormatCoo =
  GCXX_DIRECT_BACKEND_ALT(CUSPARSE_FORMAT_COO, HIPSPARSE_FORMAT_COO);
inline constexpr deviceSparseFormat_t deviceSparseFormatCsc =
  GCXX_DIRECT_BACKEND_ALT(CUSPARSE_FORMAT_CSC, HIPSPARSE_FORMAT_CSC);

inline constexpr deviceSparseOrder_t deviceSparseOrderRow =
  GCXX_DIRECT_BACKEND_ALT(CUSPARSE_ORDER_ROW, HIPSPARSE_ORDER_ROW);
inline constexpr deviceSparseOrder_t deviceSparseOrderCol =
  GCXX_DIRECT_BACKEND_ALT(CUSPARSE_ORDER_COL, HIPSPARSE_ORDER_COL);

inline constexpr deviceValueType_t deviceValueTypeF32 =
  GCXX_DIRECT_BACKEND_ALT(CUDA_R_32F, HIP_R_32F);
inline constexpr deviceValueType_t deviceValueTypeF64 =
  GCXX_DIRECT_BACKEND_ALT(CUDA_R_64F, HIP_R_64F);

GCXX_NAMESPACE_MAIN_DRIVER_END()

#endif
