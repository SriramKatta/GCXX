// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_SPARSE_DESCR_SP_MAT_DESCR_HPP_
#define GCXX_SPARSE_DESCR_SP_MAT_DESCR_HPP_

#include <cstdint>
#include <utility>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_sparse.hpp>
#include <gcxx/runtime_backend/backend_sparse_handles.hpp>
#include <gcxx/sparse/descr/sp_mat_descr_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

class SpMatDescr : public SpMatDescrView {
 public:
  // Format-specific factories.
  GCXX_FH static auto from_csr(
    std::int64_t rows, std::int64_t cols, std::int64_t nnz, void* csrRowOffsets,
    void* csrColInd, void* csrValues,
    driver::deviceSparseIndexType_t csrRowOffsetsType,
    driver::deviceSparseIndexType_t csrColIndType,
    driver::deviceSparseIndexBase_t indexBase,
    driver::deviceValueType_t valueType) -> SpMatDescr;

  GCXX_FH static auto from_coo(
    std::int64_t rows, std::int64_t cols, std::int64_t nnz, void* cooRowInd,
    void* cooColInd, void* cooValues,
    driver::deviceSparseIndexType_t cooIdxType,
    driver::deviceSparseIndexBase_t indexBase,
    driver::deviceValueType_t valueType) -> SpMatDescr;

  GCXX_FH static auto from_csc(
    std::int64_t rows, std::int64_t cols, std::int64_t nnz, void* cscColOffsets,
    void* cscRowInd, void* cscValues,
    driver::deviceSparseIndexType_t cscColOffsetsType,
    driver::deviceSparseIndexType_t cscRowIndType,
    driver::deviceSparseIndexBase_t indexBase,
    driver::deviceValueType_t valueType) -> SpMatDescr;

  GCXX_FH static auto from_native_descriptor(
    driver::deviceSpMatDescr_t descr) noexcept -> SpMatDescr;

  GCXX_FH ~SpMatDescr() noexcept;

  SpMatDescr(int)            = delete;
  SpMatDescr(std::nullptr_t) = delete;

  SpMatDescr(const SpMatDescr&)            = delete;
  SpMatDescr& operator=(const SpMatDescr&) = delete;

  GCXX_FH SpMatDescr(SpMatDescr&& other) noexcept;

  GCXX_FH auto operator=(SpMatDescr&& other) noexcept -> SpMatDescr&;

  GCXX_FH auto release() noexcept -> SpMatDescrView;

 private:
  GCXX_FH explicit SpMatDescr(driver::deviceSpMatDescr_t descr) noexcept;
};

GCXX_NAMESPACE_MAIN_END()


#include <gcxx/sparse/details/descr/sp_mat_descr.inl>


#endif
