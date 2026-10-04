// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_GRAPH_GRAPH_EXEC_VIEW_INL_
#define GCXX_RUNTIME_DETAILS_GRAPH_GRAPH_EXEC_VIEW_INL_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/graph/graph_exec_view.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_FHC
GraphExecView::GraphExecView(deviceGraphExec_t rawExec) : m_exec(rawExec) {}

GCXX_FHC auto GraphExecView::getRawHandle() const -> deviceGraphExec_t {
  return m_exec;
}

GCXX_FH auto GraphExecView::launch(
  const StreamView& stream = StreamView::Null()) const -> void {
  driver::graphLaunch(m_exec, stream.getRawHandle());
}

GCXX_FH auto GraphExecView::upload(const StreamView& stream) const -> void {
  driver::graphUpload(m_exec, stream.getRawHandle());
}

GCXX_NAMESPACE_MAIN_END()

#endif
