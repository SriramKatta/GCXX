#pragma once
#ifndef GCXX_RUNTIME_DETAILS_GRAPH_VIEW_INL_
#define GCXX_RUNTIME_DETAILS_GRAPH_VIEW_INL_

#include <gcxx/macros/define_macros.hpp>
#include <gcxx/runtime/graph/graph_view.hpp>
#include <gcxx/runtime/runtime_error.hpp>


GCXX_NAMESPACE_MAIN_BEGIN

GCXX_FHC GraphView::GraphView(deviceGraph_t rawgraph) : graph_(rawgraph){}

GCXX_FHC auto GraphView::getRawGraph() -> deviceGraph_t {
  return graph_;
}

GCXX_NAMESPACE_MAIN_END

#include <gcxx/macros/undefine_macros.hpp>

#endif