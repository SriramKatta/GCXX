// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_STREAM_STREAM_CAPTURE_INL_
#define GCXX_RUNTIME_DETAILS_STREAM_STREAM_CAPTURE_INL_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/graph/graph.hpp>
#include <gcxx/runtime/graph/graph_view.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

struct CaptureInfo {
  flags::streamCaptureStatus status{};
  unsigned long long Unique_ID{};
  GraphView graph{};
  const GraphView::deviceGraphNode_t* pDependencies{};
  std::size_t pDependenciescount{};
};

GCXX_FH auto StreamView::beginCapture(
  const flags::streamCaptureMode createflag) const -> void {
  driver::streamBeginCapture(
    m_stream, static_cast<driver::deviceStreamCaptureMode_t>(createflag));
}

GCXX_FH auto StreamView::beginCaptureToGraph(
  GraphView& graph_view,
  const flags::streamCaptureMode createflag) const -> void {
  driver::streamBeginCaptureToGraph(
    m_stream, graph_view.getRawHandle(), /*dependencies*/ nullptr,
    /*dependencyData*/ nullptr, /*numDependencies*/ 0,
    static_cast<driver::deviceStreamCaptureMode_t>(createflag));
}

GCXX_FH auto StreamView::endCapture() const -> Graph {
  const auto pgraph = driver::streamEndCapture(m_stream);
  return Graph::createFromRaw(pgraph);
}

GCXX_FH auto StreamView::endCaptureToGraph(const GraphView& graph = {}) const
  -> void {
  const auto pgraph = driver::streamEndCapture(m_stream);
  // Assert that the returned graph is indeed the same as the one we passed in
  assert(pgraph == graph.getRawHandle() &&
         "endCapture returned unexpected graph handle");
  (void)pgraph;  // Silence unused variable warning in release builds
}

#if GCXX_CUDA_MODE()
GCXX_FH auto StreamView::isCapturing() const
  -> gcxx::flags::streamCaptureStatus {
  driver::deviceStreamCaptureStatus_t status{};
  driver::streamIsCapturing(m_stream, &status);
  return flags::to_streamCaptureStatus(status);
}

GCXX_FH auto StreamView::getCaptureInfo() const -> CaptureInfo {
  driver::deviceStreamCaptureStatus_t status{};
  unsigned long long id{};
  GraphView::deviceGraph_t graph{};
  const GraphView::deviceGraphNode_t* pDependencies = nullptr;
  std::size_t numdeps                               = 0;

  driver::streamGetCaptureInfo(m_stream, &status, &id, &graph, &pDependencies,
                               nullptr, &numdeps);

  return {flags::to_streamCaptureStatus(status), id, GraphView(graph),
          pDependencies, numdeps};
}

GCXX_FH auto StreamView::updateCaptureDependencies(
  flags::StreamUpdateCaptureDependencies flag, deviceGraphNode_t* nodes,
  std::size_t numdeps) const -> void {
  driver::streamUpdateCaptureDependencies(m_stream, nodes, nullptr, numdeps,
                                          static_cast<details_::flag_t>(flag));
}
#endif

GCXX_NAMESPACE_MAIN_END()

#endif
