// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_GRAPH_GRAPH_EXEC_HPP_
#define GCXX_RUNTIME_GRAPH_GRAPH_EXEC_HPP_

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/graph/graph_exec_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

class Graph;
class GraphView;

class GraphExec : public GraphExecView {
 private:
  GCXX_FH GraphExec(deviceGraphExec_t exec) GCXX_NOEXCEPT
      : GraphExecView(exec) {}

 public:
  GCXX_FH auto destroy() -> void;

  GCXX_FH GraphExec() GCXX_NOEXCEPT : GraphExecView() {}

  GCXX_FH explicit GraphExec(const GraphView& graph);

  GCXX_FH static auto create(const GraphView& graph) -> GraphExec;

  GCXX_FH static auto createFromRaw(deviceGraphExec_t exec) -> GraphExec;

  GCXX_FH ~GraphExec() GCXX_NOEXCEPT;

  GraphExec(const GraphExec&)            = delete;
  GraphExec& operator=(const GraphExec&) = delete;

  GCXX_FH GraphExec(GraphExec&& other) GCXX_NOEXCEPT;
  GCXX_FH auto operator=(GraphExec&& other) GCXX_NOEXCEPT->GraphExec&;

  GCXX_FH auto release() GCXX_NOEXCEPT -> GraphExecView;

  GCXX_FH auto update(const GraphView& graph) -> void;
};

GCXX_NAMESPACE_MAIN_END()

#include <gcxx/runtime/details/graph/graph_exec.inl>


#endif
