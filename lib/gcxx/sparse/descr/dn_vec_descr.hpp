// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_SPARSE_DESCR_DN_VEC_DESCR_HPP_
#define GCXX_SPARSE_DESCR_DN_VEC_DESCR_HPP_

#include <cstdint>
#include <utility>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_sparse.hpp>
#include <gcxx/runtime_backend/backend_sparse_handles.hpp>
#include <gcxx/sparse/descr/dn_vec_descr_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

class DnVecDescr : public DnVecDescrView {
 public:
  GCXX_FH static auto from_dense(std::int64_t size, void* values,
                                 driver::deviceValueType_t valueType)
    -> DnVecDescr;

  GCXX_FH static auto from_native_descriptor(
    driver::deviceDnVecDescr_t descr) noexcept -> DnVecDescr;

  GCXX_FH ~DnVecDescr() noexcept;

  DnVecDescr(int)            = delete;
  DnVecDescr(std::nullptr_t) = delete;

  DnVecDescr(const DnVecDescr&)            = delete;
  DnVecDescr& operator=(const DnVecDescr&) = delete;

  GCXX_FH DnVecDescr(DnVecDescr&& other) noexcept;

  GCXX_FH auto operator=(DnVecDescr&& other) noexcept -> DnVecDescr&;

  GCXX_FH auto release() noexcept -> DnVecDescrView;

 private:
  GCXX_FH explicit DnVecDescr(driver::deviceDnVecDescr_t descr) noexcept;
};

GCXX_NAMESPACE_MAIN_END()


#include <gcxx/sparse/details/descr/dn_vec_descr.inl>


#endif
