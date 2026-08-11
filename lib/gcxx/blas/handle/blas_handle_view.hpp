// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_HANDLE_BLAS_HANDLE_VIEW_HPP_
#define GCXX_BLAS_HANDLE_BLAS_HANDLE_VIEW_HPP_

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/stream/stream_view.hpp>
#include <gcxx/runtime_backend/backend_blas.hpp>
#include <gcxx/runtime_backend/backend_blas_handles.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

class BlasHandleView {
 protected:
  using deviceBlasHandle_t = driver::deviceBlasHandle_t;

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  deviceBlasHandle_t m_handle{nullptr};

 public:
  using raw_handle_type = driver::deviceBlasHandle_t;

  BlasHandleView()               = delete;
  BlasHandleView(int)            = delete;
  BlasHandleView(std::nullptr_t) = delete;

  GCXX_FHC explicit BlasHandleView(deviceBlasHandle_t handle) noexcept;

  // ╔════════════════════════════════════════════════════════╗
  // ║                    Stream binding                      ║
  // ╚════════════════════════════════════════════════════════╝

  GCXX_FH auto setStream(gcxx::StreamView stream) -> void;

  GCXX_FH auto getStream() const -> gcxx::StreamView;

  // ╔════════════════════════════════════════════════════════╗
  // ║                     Introspection                      ║
  // ╚════════════════════════════════════════════════════════╝

  GCXX_FH auto getVersion() const -> int;

  GCXX_FHC auto getRawHandle() const noexcept -> deviceBlasHandle_t;

  // ╔════════════════════════════════════════════════════════╗
  // ║                      Comparison                        ║
  // ╚════════════════════════════════════════════════════════╝
  GCXX_FHC auto operator==(const BlasHandleView& rhs) const noexcept -> bool;
  GCXX_FHC auto operator!=(const BlasHandleView& rhs) const noexcept -> bool;
};

GCXX_NAMESPACE_MAIN_END()


#include <gcxx/blas/details/handle/blas_handle_view.inl>


#endif
