// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_SPARSE_DETAILS_DESCR_DN_VEC_DESCR_INL_
#define GCXX_SPARSE_DETAILS_DESCR_DN_VEC_DESCR_INL_

#include <cstdint>
#include <utility>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_sparse.hpp>
#include <gcxx/runtime_backend/backend_sparse_handles.hpp>
#include <gcxx/sparse/descr/dn_vec_descr_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_FH auto DnVecDescr::from_dense(std::int64_t size, void* values,
                                    driver::deviceValueType_t valueType)
  -> DnVecDescr {
  driver::deviceDnVecDescr_t descr{};
  driver::sparseCreateDnVec(&descr, size, values, valueType);
  return DnVecDescr(descr);
}

GCXX_FH auto DnVecDescr::from_native_descriptor(
  driver::deviceDnVecDescr_t descr) noexcept -> DnVecDescr {
  return DnVecDescr(descr);
}

GCXX_FH DnVecDescr::~DnVecDescr() noexcept {
  if (m_descr != nullptr) {
    driver::sparseDestroyDnVec(m_descr);
  }
}

GCXX_FH DnVecDescr::DnVecDescr(DnVecDescr&& other) noexcept
    : DnVecDescrView(std::exchange(other.m_descr, nullptr)) {}

GCXX_FH auto DnVecDescr::operator=(DnVecDescr&& other) noexcept -> DnVecDescr& {
  if (this != &other) {
    if (m_descr != nullptr) {
      driver::sparseDestroyDnVec(m_descr);
    }
    m_descr = std::exchange(other.m_descr, nullptr);
  }
  return *this;
}

GCXX_FH auto DnVecDescr::release() noexcept -> DnVecDescrView {
  auto d  = m_descr;
  m_descr = nullptr;
  return DnVecDescrView{d};
}

GCXX_FH DnVecDescr::DnVecDescr(driver::deviceDnVecDescr_t descr) noexcept
    : DnVecDescrView(descr) {}

GCXX_NAMESPACE_MAIN_END()

#endif
