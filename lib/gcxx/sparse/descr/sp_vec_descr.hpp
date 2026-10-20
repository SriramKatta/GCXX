// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_SPARSE_DESCR_SP_VEC_DESCR_HPP_
#define GCXX_SPARSE_DESCR_SP_VEC_DESCR_HPP_

#include <cstdint>
#include <utility>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_sparse.hpp>
#include <gcxx/runtime_backend/backend_sparse_handles.hpp>
#include <gcxx/sparse/descr/sp_vec_descr_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

class SpVecDescr : public SpVecDescrView {
 public:
  GCXX_FH static auto from_sparse(
    std::int64_t size, std::int64_t nnz, void* indices, void* values,
    driver::deviceSparseIndexType_t idxType,
    driver::deviceSparseIndexBase_t indexBase,
    driver::deviceValueType_t valueType) -> SpVecDescr;

  GCXX_FH static auto from_native_descriptor(
    driver::deviceSpVecDescr_t descr) noexcept -> SpVecDescr;

  GCXX_FH ~SpVecDescr() noexcept;

  SpVecDescr(int)            = delete;
  SpVecDescr(std::nullptr_t) = delete;

  SpVecDescr(const SpVecDescr&)            = delete;
  SpVecDescr& operator=(const SpVecDescr&) = delete;

  GCXX_FH SpVecDescr(SpVecDescr&& other) noexcept;

  GCXX_FH auto operator=(SpVecDescr&& other) noexcept -> SpVecDescr&;

  GCXX_FH auto release() noexcept -> SpVecDescrView;

 private:
  GCXX_FH explicit SpVecDescr(driver::deviceSpVecDescr_t descr) noexcept;
};

GCXX_NAMESPACE_MAIN_END()


#include <gcxx/sparse/details/descr/sp_vec_descr.inl>


#endif
