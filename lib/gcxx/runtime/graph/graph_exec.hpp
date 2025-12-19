#pragma once
#ifndef GCXX_RUNTIME_GRAPH_GRAPH_EXEC_HPP_
#define GCXX_RUNTIME_GRAPH_GRAPH_EXEC_HPP_

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>

#include <gcxx/runtime/graph/graph_exec_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

class Graph;
class GraphView;

class GraphExec : public GraphExecView {
 private:
  GCXX_FH GraphExec(deviceGraphExec_t exec) GCXX_NOEXCEPT : GraphExecView(exec) {}
  GCXX_FH auto destroy() -> void;

 public:
  GCXX_FH GraphExec() GCXX_NOEXCEPT : GraphExecView() {}

  GCXX_FH explicit GraphExec(const GraphView& graph);

  GCXX_FH static auto Create(const GraphView& graph) -> GraphExec;

  GCXX_FH static auto CreateFromRaw(deviceGraphExec_t exec) -> GraphExec;

  GCXX_FH ~GraphExec() GCXX_NOEXCEPT;

  GraphExec(const GraphExec&) = delete;
  GraphExec& operator=(const GraphExec&) = delete;

  GCXX_FH GraphExec(GraphExec&& other) GCXX_NOEXCEPT;
  GCXX_FH auto operator=(GraphExec&& other) GCXX_NOEXCEPT -> GraphExec&;

  GCXX_FH auto Release() GCXX_NOEXCEPT -> GraphExecView;

  GCXX_FH auto Update(const GraphView& graph) -> void;
};

GCXX_NAMESPACE_MAIN_END

#include <gcxx/runtime/details/graph_exec.inl>

#include <gcxx/macros/undefine_macros.hpp>

#endif
