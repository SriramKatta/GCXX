// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_SPARSE_DETAILS_DESCR_SP_MAT_DESCR_INL_
#define GCXX_SPARSE_DETAILS_DESCR_SP_MAT_DESCR_INL_

#include <cstdint>
#include <utility>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_sparse.hpp>
#include <gcxx/runtime_backend/backend_sparse_handles.hpp>
#include <gcxx/sparse/descr/sp_mat_descr_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_FH auto SpMatDescr::from_csr(
  std::int64_t rows, std::int64_t cols, std::int64_t nnz, void* csrRowOffsets,
  void* csrColInd, void* csrValues,
  driver::deviceSparseIndexType_t csrRowOffsetsType,
  driver::deviceSparseIndexType_t csrColIndType,
  driver::deviceSparseIndexBase_t indexBase,
  driver::deviceValueType_t valueType) -> SpMatDescr {
  driver::deviceSpMatDescr_t descr{};
  driver::sparseCreateCsr(&descr, rows, cols, nnz, csrRowOffsets, csrColInd,
                          csrValues, csrRowOffsetsType, csrColIndType,
                          indexBase, valueType);
  return SpMatDescr(descr);
}

GCXX_FH auto SpMatDescr::from_coo(
  std::int64_t rows, std::int64_t cols, std::int64_t nnz, void* cooRowInd,
  void* cooColInd, void* cooValues, driver::deviceSparseIndexType_t cooIdxType,
  driver::deviceSparseIndexBase_t indexBase,
  driver::deviceValueType_t valueType) -> SpMatDescr {
  driver::deviceSpMatDescr_t descr{};
  driver::sparseCreateCoo(&descr, rows, cols, nnz, cooRowInd, cooColInd,
                          cooValues, cooIdxType, indexBase, valueType);
  return SpMatDescr(descr);
}

GCXX_FH auto SpMatDescr::from_csc(
  std::int64_t rows, std::int64_t cols, std::int64_t nnz, void* cscColOffsets,
  void* cscRowInd, void* cscValues,
  driver::deviceSparseIndexType_t cscColOffsetsType,
  driver::deviceSparseIndexType_t cscRowIndType,
  driver::deviceSparseIndexBase_t indexBase,
  driver::deviceValueType_t valueType) -> SpMatDescr {
  driver::deviceSpMatDescr_t descr{};
  driver::sparseCreateCsc(&descr, rows, cols, nnz, cscColOffsets, cscRowInd,
                          cscValues, cscColOffsetsType, cscRowIndType,
                          indexBase, valueType);
  return SpMatDescr(descr);
}

GCXX_FH auto SpMatDescr::from_native_descriptor(
  driver::deviceSpMatDescr_t descr) noexcept -> SpMatDescr {
  return SpMatDescr(descr);
}

GCXX_FH SpMatDescr::~SpMatDescr() noexcept {
  if (m_descr != nullptr) {
    driver::sparseDestroySpMat(m_descr);
  }
}

GCXX_FH SpMatDescr::SpMatDescr(SpMatDescr&& other) noexcept
    : SpMatDescrView(std::exchange(other.m_descr, nullptr)) {}

GCXX_FH auto SpMatDescr::operator=(SpMatDescr&& other) noexcept -> SpMatDescr& {
  if (this != &other) {
    if (m_descr != nullptr) {
      driver::sparseDestroySpMat(m_descr);
    }
    m_descr = std::exchange(other.m_descr, nullptr);
  }
  return *this;
}

GCXX_FH auto SpMatDescr::release() noexcept -> SpMatDescrView {
  auto d  = m_descr;
  m_descr = nullptr;
  return SpMatDescrView{d};
}

GCXX_FH SpMatDescr::SpMatDescr(driver::deviceSpMatDescr_t descr) noexcept
    : SpMatDescrView(descr) {}

GCXX_NAMESPACE_MAIN_END()

#endif
