// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_SPARSE_DESCR_DN_MAT_DESCR_HPP_
#define GCXX_SPARSE_DESCR_DN_MAT_DESCR_HPP_

#include <cstdint>
#include <utility>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_sparse.hpp>
#include <gcxx/runtime_backend/backend_sparse_handles.hpp>
#include <gcxx/sparse/descr/dn_mat_descr_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

class DnMatDescr : public DnMatDescrView {
 public:
  GCXX_FH static auto from_dense(
    std::int64_t rows, std::int64_t cols, std::int64_t ld, void* values,
    driver::deviceValueType_t valueType,
    driver::deviceSparseOrder_t order) -> DnMatDescr;

  GCXX_FH static auto from_native_descriptor(
    driver::deviceDnMatDescr_t descr) noexcept -> DnMatDescr;

  GCXX_FH ~DnMatDescr() noexcept;

  DnMatDescr(int)            = delete;
  DnMatDescr(std::nullptr_t) = delete;

  DnMatDescr(const DnMatDescr&)            = delete;
  DnMatDescr& operator=(const DnMatDescr&) = delete;

  GCXX_FH DnMatDescr(DnMatDescr&& other) noexcept;

  GCXX_FH auto operator=(DnMatDescr&& other) noexcept -> DnMatDescr&;

  GCXX_FH auto release() noexcept -> DnMatDescrView;

 private:
  GCXX_FH explicit DnMatDescr(driver::deviceDnMatDescr_t descr) noexcept;
};

GCXX_NAMESPACE_MAIN_END()


#include <gcxx/sparse/details/descr/dn_mat_descr.inl>


#endif
