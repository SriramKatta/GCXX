#pragma once
#ifndef GCXX_RUNTIME_DETAILS_GRAPH_GRAPH_EXEC_VIEW_INL_
#define GCXX_RUNTIME_DETAILS_GRAPH_GRAPH_EXEC_VIEW_INL_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/graph/graph_exec_view.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

GCXX_FHC GraphExecView::GraphExecView(deviceGraphExec_t rawExec)
    : exec_(rawExec) {}

GCXX_FHC auto GraphExecView::getRawExec() const -> deviceGraphExec_t {
  return exec_;
}

GCXX_FHC GraphExecView::operator deviceGraphExec_t() const GCXX_NOEXCEPT {
  return exec_;
}

GCXX_FH auto GraphExecView::Launch(
  const StreamView& stream = details_::NULL_STREAM) const -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphLaunch, "Failed to launch graph", exec_,
                         stream.getRawStream());
}

GCXX_FH auto GraphExecView::Upload(const StreamView& stream) const -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphUpload, "Failed to upload graph", exec_,
                         stream.getRawStream());
}

GCXX_NAMESPACE_MAIN_END

#include <gcxx/macros/undefine_macros.hpp>

#endif
