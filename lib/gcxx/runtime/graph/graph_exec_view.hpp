// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_GRAPH_GRAPH_EXEC_VIEW_HPP_
#define GCXX_RUNTIME_GRAPH_GRAPH_EXEC_VIEW_HPP_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_graph.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

class StreamView;

class GraphExecView {
 public:
  using deviceGraphExec_t = driver::deviceGraphExec_t;

 protected:
  deviceGraphExec_t exec_{driver::INVALID_GRAPH_EXEC};  // NOLINT

 public:
  GCXX_FHC GraphExecView() = default;
  GCXX_FHC GraphExecView(deviceGraphExec_t rawExec);
  GCXX_FHC auto getRawExec() const -> deviceGraphExec_t;
  GCXX_FHC operator deviceGraphExec_t() const GCXX_NOEXCEPT;

  GCXX_FH auto Launch(const StreamView& stream) const -> void;
  GCXX_FH auto Upload(const StreamView& stream) const -> void;
};

GCXX_NAMESPACE_MAIN_END()

#include <gcxx/runtime/details/graph/graph_exec_view.inl>

#include <gcxx/macros/undefine_macros.hpp>
#endif