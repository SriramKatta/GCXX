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
  using raw_handle_type   = driver::deviceGraphExec_t;

  GCXX_FHC GraphExecView() = default;
  GCXX_FHC GraphExecView(deviceGraphExec_t rawExec);
  GCXX_FHC auto getRawHandle() const -> deviceGraphExec_t;

  GCXX_FH auto Launch(const StreamView& stream) const -> void;
  GCXX_FH auto Upload(const StreamView& stream) const -> void;


 protected:
  deviceGraphExec_t m_exec{driver::INVALID_GRAPH_EXEC};  // NOLINT
};

GCXX_NAMESPACE_MAIN_END()

#include <gcxx/runtime/details/graph/graph_exec_view.inl>


#endif