// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_GRAPH_NODES_GRAPH_NODE_VIEW_INL_
#define GCXX_RUNTIME_DETAILS_GRAPH_NODES_GRAPH_NODE_VIEW_INL_

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/flags/graph_flags.hpp>


GCXX_NAMESPACE_MAIN_BEGIN()


#if GCXX_CUDA_VERSION_GREATER_EQUAL(13, 1, 0)
GCXX_FH auto GraphNodeView::geLocalId() -> unsigned int {
  return driver::graphNodeGetLocalId(node_);
}

GCXX_FH auto GraphNodeView::geToolsId() -> unsigned long long {
  return driver::graphNodeGetToolsId(node_);
}
#endif

GCXX_FH auto GraphNodeView::getType() -> flags::graphNodeType {
  return static_cast<flags::graphNodeType>(driver::graphNodeGetType(node_));
}

GCXX_NAMESPACE_MAIN_END()

#endif