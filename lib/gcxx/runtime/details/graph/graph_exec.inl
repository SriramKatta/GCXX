#pragma once
#ifndef GCXX_RUNTIME_DETAILS_GRAPH_GRAPH_EXEC_INL_
#define GCXX_RUNTIME_DETAILS_GRAPH_GRAPH_EXEC_INL_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/graph/graph_exec.hpp>
#include <gcxx/runtime/graph/graph_view.hpp>

#include <utility>

GCXX_NAMESPACE_MAIN_BEGIN

GCXX_FH auto GraphExec::Create(const GraphView& graph) -> GraphExec {
  return GraphExec{graph};
}

GCXX_FH GraphExec::GraphExec(const GraphView& graph)
    : GraphExecView(details_::INVALID_GRAPH_EXEC) {
  GCXX_SAFE_RUNTIME_CALL(GraphInstantiate, "Failed to instantiate the graph",
                         &exec_, graph.getRawGraph(), nullptr, nullptr, 0);
}

GCXX_FH auto GraphExec::CreateFromRaw(deviceGraphExec_t exec) -> GraphExec {
  return GraphExec{exec};
}

GCXX_FH auto GraphExec::destroy() -> void {
  if (exec_ != details_::INVALID_GRAPH_EXEC) {
    GCXX_SAFE_RUNTIME_CALL(GraphExecDestroy, "Failed to destroy graph exec",
                           exec_);
  }
}

GCXX_FH GraphExec::~GraphExec() GCXX_NOEXCEPT {
  destroy();
}

GCXX_FH GraphExec::GraphExec(GraphExec&& other) GCXX_NOEXCEPT
    : GraphExecView(std::exchange(other.exec_, details_::INVALID_GRAPH_EXEC)) {}

GCXX_FH auto GraphExec::operator=(GraphExec&& other) GCXX_NOEXCEPT->GraphExec& {
  if (this != &other) {
    destroy();
    exec_ = std::exchange(other.exec_, details_::INVALID_GRAPH_EXEC);
  }
  return *this;
}

GCXX_FH auto GraphExec::Release() GCXX_NOEXCEPT->GraphExecView {
  auto oldExec = exec_;
  exec_        = details_::INVALID_GRAPH_EXEC;
  return GraphExecView{oldExec};
}

GCXX_FH auto GraphExec::Update(const GraphView& graph) -> void {
  // TODO hip needs the 4th arg and the last arg has different datatype names in
  // cuda and hip
  // GCXX_RUNTIME_BACKEND(GraphExecUpdateResultInfo) updateResult;
  GCXX_SAFE_RUNTIME_CALL(GraphExecUpdate, "Failed to update graph exec", exec_,
                         graph.getRawGraph(),
#if GCXX_HIP_MODE
                         NULL,
#endif
                         nullptr);
}

GCXX_NAMESPACE_MAIN_END

#endif
