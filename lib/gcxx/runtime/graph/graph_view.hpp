#pragma once
#ifndef GCXX_RUNTIME_GRAPH_GRAPH_VIEW_HPP_
#define GCXX_RUNTIME_GRAPH_GRAPH_VIEW_HPP_

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>

#include <gcxx/runtime/flags/graph_flags.hpp>


GCXX_NAMESPACE_MAIN_DETAILS_BEGIN
using deviceGraph_t = GCXX_RUNTIME_BACKEND(Graph_t);
GCXX_NAMESPACE_MAIN_DETAILS_END


GCXX_NAMESPACE_MAIN_BEGIN

class GraphView {
 protected:
  using deviceGraph_t = details_::deviceGraph_t;
  deviceGraph_t graph_{nullptr};

 public:
  GCXX_FHC GraphView() = default;
  GCXX_FHC GraphView(deviceGraph_t rawgraph);
  GCXX_FHC auto getRawGraph() -> deviceGraph_t;
};

GCXX_NAMESPACE_MAIN_END

#include <gcxx/runtime/details/graph_view.inl>

#include <gcxx/macros/undefine_macros.hpp>
#endif