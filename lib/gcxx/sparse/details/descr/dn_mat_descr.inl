// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_SPARSE_DETAILS_DESCR_DN_MAT_DESCR_INL_
#define GCXX_SPARSE_DETAILS_DESCR_DN_MAT_DESCR_INL_

#include <cstdint>
#include <utility>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_sparse.hpp>
#include <gcxx/runtime_backend/backend_sparse_handles.hpp>
#include <gcxx/sparse/descr/dn_mat_descr_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_FH auto DnMatDescr::from_dense(
  std::int64_t rows, std::int64_t cols, std::int64_t ld, void* values,
  driver::deviceValueType_t valueType,
  driver::deviceSparseOrder_t order) -> DnMatDescr {
  driver::deviceDnMatDescr_t descr{};
  driver::sparseCreateDnMat(&descr, rows, cols, ld, values, valueType, order);
  return DnMatDescr(descr);
}

GCXX_FH auto DnMatDescr::from_native_descriptor(
  driver::deviceDnMatDescr_t descr) noexcept -> DnMatDescr {
  return DnMatDescr(descr);
}

GCXX_FH DnMatDescr::~DnMatDescr() noexcept {
  if (m_descr != nullptr) {
    driver::sparseDestroyDnMat(m_descr);
  }
}

GCXX_FH DnMatDescr::DnMatDescr(DnMatDescr&& other) noexcept
    : DnMatDescrView(std::exchange(other.m_descr, nullptr)) {}

GCXX_FH auto DnMatDescr::operator=(DnMatDescr&& other) noexcept -> DnMatDescr& {
  if (this != &other) {
    if (m_descr != nullptr) {
      driver::sparseDestroyDnMat(m_descr);
    }
    m_descr = std::exchange(other.m_descr, nullptr);
  }
  return *this;
}

GCXX_FH auto DnMatDescr::release() noexcept -> DnMatDescrView {
  auto d  = m_descr;
  m_descr = nullptr;
  return DnMatDescrView{d};
}

GCXX_FH DnMatDescr::DnMatDescr(driver::deviceDnMatDescr_t descr) noexcept
    : DnMatDescrView(descr) {}

GCXX_NAMESPACE_MAIN_END()

#endif
