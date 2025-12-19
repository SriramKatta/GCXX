#pragma once
#ifndef GCXX_API_RUNTIME_GRAPH_GRAPH_HPP_
#define GCXX_API_RUNTIME_GRAPH_GRAPH_HPP_

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>

#include <gcxx/runtime/graph/graph_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

class Graph : public GraphView {
 private:
  GCXX_FH Graph(deviceGraph_t graph) GCXX_NOEXCEPT : GraphView(graph) {}

 public:
  GCXX_FH Graph(const flags::graphCreate createFlag = flags::graphCreate::none)
    GCXX_NOEXCEPT;
  GCXX_FH ~Graph() GCXX_NOEXCEPT;
  GCXX_FH static auto CreateFromRaw(deviceGraph_t graph) -> Graph;
};

GCXX_NAMESPACE_MAIN_END

#include <gcxx/runtime/details/graph.inl>

#include <gcxx/macros/undefine_macros.hpp>

#endif
