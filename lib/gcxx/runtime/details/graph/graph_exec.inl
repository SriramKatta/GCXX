// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_GRAPH_GRAPH_EXEC_INL_
#define GCXX_RUNTIME_DETAILS_GRAPH_GRAPH_EXEC_INL_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/graph/graph_exec.hpp>
#include <gcxx/runtime/graph/graph_view.hpp>

#include <utility>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_FH auto GraphExec::Create(const GraphView& graph) -> GraphExec {
  return GraphExec{graph};
}

GCXX_FH GraphExec::GraphExec(const GraphView& graph)
    : GraphExecView(driver::graphInstantiate(graph.getRawGraph())) {}

GCXX_FH auto GraphExec::CreateFromRaw(deviceGraphExec_t exec) -> GraphExec {
  return GraphExec{exec};
}

GCXX_FH auto GraphExec::destroy() -> void {
  if (m_exec != driver::INVALID_GRAPH_EXEC) {
    driver::graphExecDestroy(m_exec);
  }
}

GCXX_FH GraphExec::~GraphExec() GCXX_NOEXCEPT {
  destroy();
}

GCXX_FH GraphExec::GraphExec(GraphExec&& other) GCXX_NOEXCEPT
    : GraphExecView(std::exchange(other.m_exec, driver::INVALID_GRAPH_EXEC)) {}

GCXX_FH auto GraphExec::operator=(GraphExec&& other)
  GCXX_NOEXCEPT -> GraphExec& {
  if (this != &other) {
    destroy();
    m_exec = std::exchange(other.m_exec, driver::INVALID_GRAPH_EXEC);
  }
  return *this;
}

GCXX_FH auto GraphExec::Release() GCXX_NOEXCEPT -> GraphExecView {
  auto oldExec = m_exec;
  m_exec       = driver::INVALID_GRAPH_EXEC;
  return GraphExecView{oldExec};
}

GCXX_FH auto GraphExec::Update(const GraphView& graph) -> void {
  driver::graphExecUpdate(m_exec, graph.getRawGraph());
}

GCXX_NAMESPACE_MAIN_END()

#endif
