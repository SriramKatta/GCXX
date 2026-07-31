// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_STREAM_STREAM_VIEW_INL_
#define GCXX_RUNTIME_DETAILS_STREAM_STREAM_VIEW_INL_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/graph/graph.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

struct CaptureInfo {
  flags::streamCaptureStatus status{};
  unsigned long long Unique_ID{};
  GraphView graph{};
  const GraphView::deviceGraphNode_t* pDependencies{};
  std::size_t pDependenciescount{};
};

GCXX_FHC
StreamView::StreamView(deviceStream_t rawStream) GCXX_NOEXCEPT
    : m_stream(rawStream) {}

GCXX_FHC auto StreamView::getRawStream() GCXX_CONST_NOEXCEPT -> deviceStream_t {
  return m_stream;
}

GCXX_FHC StreamView::operator deviceStream_t() GCXX_CONST_NOEXCEPT {
  return getRawStream();
}

GCXX_FH auto StreamView::HasPendingWork() -> bool {
  const auto err = driver::streamQueryNothrow(m_stream);
  return !details_::nonFatalErrorQuery(err);
}

GCXX_FH auto StreamView::Synchronize() const -> void {
  driver::streamSynchronize(m_stream);
}

GCXX_FH auto StreamView::WaitOnEvent(const EventView& event,
                                     flags::eventWait waitFlag) const -> void {
  driver::StreamWaitEvent(this->m_stream, event.getRawEvent(),
                          static_cast<details_::flag_t>(waitFlag));
}

GCXX_FH auto StreamView::BeginCapture(
  const flags::streamCaptureMode createflag) const -> void {
  driver::streamBeginCapture(
    m_stream, static_cast<driver::deviceStreamCaptureMode_t>(createflag));
}

GCXX_FH auto StreamView::BeginCaptureToGraph(
  GraphView& graph_view, const flags::streamCaptureMode createflag) const
  -> void {
  driver::streamBeginCaptureToGraph(
    m_stream, graph_view.getRawGraph(), nullptr, nullptr, 0,
    static_cast<driver::deviceStreamCaptureMode_t>(createflag));
}

GCXX_FH auto StreamView::EndCapture() const -> Graph {
  const auto pgraph = driver::streamEndCapture(m_stream);
  return Graph::CreateFromRaw(pgraph);
}

GCXX_FH auto StreamView::EndCaptureToGraph(const GraphView& graph = {}) const
  -> void {
  // When using BeginCaptureToGraph, the capture happens into the existing
  // graph, so the returned handle from EndCapture is the same as
  // graph.getRawGraph(). We just need to call EndCapture to finalize the
  // capture.
  const auto pgraph = driver::streamEndCapture(m_stream);
  // Assert that the returned graph is indeed the same as the one we passed in
  assert(pgraph == graph.getRawGraph() &&
         "EndCapture returned unexpected graph handle");
  (void)pgraph;  // Silence unused variable warning in release builds
}
#if GCXX_CUDA_MODE()
GCXX_FH auto StreamView::IsCapturing() const
  -> gcxx::flags::streamCaptureStatus {
  driver::deviceStreamCaptureStatus_t status{};
  driver::streamIsCapturing(m_stream, &status);
  return flags::to_streamCaptureStatus(status);
}

GCXX_FH auto StreamView::GetCaptureInfo() const -> CaptureInfo {
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

GCXX_FH auto StreamView::UpdateCaptureDependencies(
  flags::StreamUpdateCaptureDependencies flag, deviceGraphNode_t* nodes,
  std::size_t numdeps) const -> void {
  driver::streamUpdateCaptureDependencies(m_stream, nodes, nullptr, numdeps,
                                          static_cast<details_::flag_t>(flag));
}
#endif
GCXX_NAMESPACE_MAIN_END()


#endif