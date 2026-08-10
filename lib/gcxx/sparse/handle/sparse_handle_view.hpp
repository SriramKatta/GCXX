// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_SPARSE_HANDLE_SPARSE_HANDLE_VIEW_HPP_
#define GCXX_SPARSE_HANDLE_SPARSE_HANDLE_VIEW_HPP_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>
#include <gcxx/runtime_backend/backend_sparse.hpp>
#include <gcxx/runtime_backend/backend_sparse_handles.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

class SparseHandleView {
 protected:
  using deviceSparseHandle_t = driver::deviceSparseHandle_t;

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  deviceSparseHandle_t m_handle{nullptr};

 public:
  SparseHandleView()               = delete;
  SparseHandleView(int)            = delete;
  SparseHandleView(std::nullptr_t) = delete;

  GCXX_FHC explicit SparseHandleView(deviceSparseHandle_t handle) noexcept;

  // ╔════════════════════════════════════════════════════════╗
  // ║                    Stream binding                      ║
  // ╚════════════════════════════════════════════════════════╝

  GCXX_FH auto setStream(gcxx::StreamView stream) -> void;

  GCXX_FH auto getStream() const -> gcxx::StreamView;

  // ╔════════════════════════════════════════════════════════╗
  // ║                     Introspection                      ║
  // ╚════════════════════════════════════════════════════════╝

  GCXX_FH auto getVersion() const -> int;

  GCXX_FHC auto getHandle() const noexcept -> deviceSparseHandle_t;

  // ╔════════════════════════════════════════════════════════╗
  // ║                      Comparison                        ║
  // ╚════════════════════════════════════════════════════════╝
  GCXX_FHC auto operator==(const SparseHandleView& rhs) const noexcept -> bool;
  GCXX_FHC auto operator!=(const SparseHandleView& rhs) const noexcept -> bool;
};

GCXX_NAMESPACE_MAIN_END()


#include <gcxx/sparse/details/handle/sparse_handle_view.inl>


#endif
