// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_HANDLE_BLAS_HANDLE_HPP_
#define GCXX_BLAS_HANDLE_BLAS_HANDLE_HPP_

#include <utility>

#include <gcxx/blas/handle/blas_handle_view.hpp>
#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_blas.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()


class BlasHandle : public BlasHandleView {
 public:
  BlasHandle();

  GCXX_FH ~BlasHandle() noexcept;

  BlasHandle(int)            = delete;
  BlasHandle(std::nullptr_t) = delete;

  BlasHandle(const BlasHandle&)            = delete;
  BlasHandle& operator=(const BlasHandle&) = delete;

  GCXX_FH BlasHandle(BlasHandle&& other) noexcept;

  GCXX_FH auto operator=(BlasHandle&& other) noexcept -> BlasHandle&;

  GCXX_FH static auto from_native_handle(
    driver::deviceBlasHandle_t handle) noexcept -> BlasHandle;

  GCXX_FH auto release() noexcept -> BlasHandleView;

  // Safe to call multiple times; also invoked by the destructor.
  GCXX_FH auto destroy() noexcept -> void;

 private:
  GCXX_FH explicit BlasHandle(driver::deviceBlasHandle_t handle) noexcept;
};

GCXX_NAMESPACE_MAIN_BLAS_END()


#include <gcxx/blas/details/handle/blas_handle.inl>


#endif
