// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_SPARSE_HANDLE_SPARSE_HANDLE_HPP_
#define GCXX_SPARSE_HANDLE_SPARSE_HANDLE_HPP_

#include <utility>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_sparse.hpp>
#include <gcxx/sparse/handle/sparse_handle_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()


class SparseHandle : public SparseHandleView {
 public:
  SparseHandle();

  GCXX_FH ~SparseHandle() noexcept;

  SparseHandle(int)            = delete;
  SparseHandle(std::nullptr_t) = delete;

  SparseHandle(const SparseHandle&)            = delete;
  SparseHandle& operator=(const SparseHandle&) = delete;

  GCXX_FH SparseHandle(SparseHandle&& other) noexcept;

  GCXX_FH auto operator=(SparseHandle&& other) noexcept -> SparseHandle&;

  GCXX_FH static auto from_native_handle(
    driver::deviceSparseHandle_t handle) noexcept -> SparseHandle;

  GCXX_FH auto release() noexcept -> SparseHandleView;

  // Safe to call multiple times; also invoked by the destructor.
  GCXX_FH auto destroy() noexcept -> void;

 private:
  GCXX_FH explicit SparseHandle(driver::deviceSparseHandle_t handle) noexcept;
};

GCXX_NAMESPACE_MAIN_END()


#include <gcxx/sparse/details/handle/sparse_handle.inl>


#endif
