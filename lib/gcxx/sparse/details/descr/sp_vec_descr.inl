// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_SPARSE_DETAILS_DESCR_SP_VEC_DESCR_INL_
#define GCXX_SPARSE_DETAILS_DESCR_SP_VEC_DESCR_INL_

#include <cstdint>
#include <utility>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_sparse.hpp>
#include <gcxx/runtime_backend/backend_sparse_handles.hpp>
#include <gcxx/sparse/descr/sp_vec_descr_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_FH auto SpVecDescr::from_sparse(
  std::int64_t size, std::int64_t nnz, void* indices, void* values,
  driver::deviceSparseIndexType_t idxType,
  driver::deviceSparseIndexBase_t indexBase,
  driver::deviceValueType_t valueType) -> SpVecDescr {
  driver::deviceSpVecDescr_t descr{};
  driver::sparseCreateSpVec(&descr, size, nnz, indices, values, idxType,
                            indexBase, valueType);
  return SpVecDescr(descr);
}

GCXX_FH auto SpVecDescr::from_native_descriptor(
  driver::deviceSpVecDescr_t descr) noexcept -> SpVecDescr {
  return SpVecDescr(descr);
}

GCXX_FH SpVecDescr::~SpVecDescr() noexcept {
  if (m_descr != nullptr) {
    driver::sparseDestroySpVec(m_descr);
  }
}

GCXX_FH SpVecDescr::SpVecDescr(SpVecDescr&& other) noexcept
    : SpVecDescrView(std::exchange(other.m_descr, nullptr)) {}

GCXX_FH auto SpVecDescr::operator=(SpVecDescr&& other) noexcept -> SpVecDescr& {
  if (this != &other) {
    if (m_descr != nullptr) {
      driver::sparseDestroySpVec(m_descr);
    }
    m_descr = std::exchange(other.m_descr, nullptr);
  }
  return *this;
}

GCXX_FH auto SpVecDescr::release() noexcept -> SpVecDescrView {
  auto d  = m_descr;
  m_descr = nullptr;
  return SpVecDescrView{d};
}

GCXX_FH SpVecDescr::SpVecDescr(driver::deviceSpVecDescr_t descr) noexcept
    : SpVecDescrView(descr) {}

GCXX_NAMESPACE_MAIN_END()

#endif
