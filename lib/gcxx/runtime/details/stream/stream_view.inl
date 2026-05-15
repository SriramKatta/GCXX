// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_STREAM_STREAM_VIEW_INL_
#define GCXX_RUNTIME_DETAILS_STREAM_STREAM_VIEW_INL_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/graph/graph.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

struct CaptureInfo {
  flags::streamCaptureStatus status{};
  unsigned long long Unique_ID{};
  GraphView graph{};
  const GraphView::deviceGraphNode_t* pDependencies{};
  std::size_t pDependenciescount{};
};

GCXX_FHC StreamView::StreamView(deviceStream_t rawStream) GCXX_NOEXCEPT
    : stream_(rawStream) {}

GCXX_FH constexpr auto StreamView::getRawStream()
  GCXX_CONST_NOEXCEPT -> deviceStream_t {
  return stream_;
}

GCXX_FH constexpr StreamView::operator deviceStream_t() GCXX_CONST_NOEXCEPT {
  return getRawStream();
}

GCXX_FH auto StreamView::HasPendingWork() -> bool {
  const auto err = driver::streamQueryNothrow(stream_);
  return !details_::nonFatalErrorQuery(err);
}

GCXX_FH auto StreamView::Synchronize() const -> void {
  driver::streamSynchronize(stream_);
}

GCXX_FH auto StreamView::WaitOnEvent(
  const EventView& event, const flags::eventWait waitFlag) const -> void {
  driver::StreamWaitEvent(this->stream_, event.getRawEvent(),
                          static_cast<details_::flag_t>(waitFlag));
}

GCXX_FH auto StreamView::BeginCapture(const flags::streamCaptureMode createflag)
  -> void {
  GCXX_SAFE_RUNTIME_CALL(
    StreamBeginCapture, "Failed to begin Stream Capture", this->getRawStream(),
    static_cast<GCXX_RUNTIME_BACKEND(StreamCaptureMode)>(createflag));
}

GCXX_FH auto StreamView::BeginCaptureToGraph(
  GraphView& graph_view, const flags::streamCaptureMode createflag) -> void {
  GCXX_SAFE_RUNTIME_CALL(
    StreamBeginCaptureToGraph, "Failed to begin Stream Capture to graph",
    this->getRawStream(), graph_view.getRawGraph(), nullptr, nullptr, 0,
    static_cast<GCXX_RUNTIME_BACKEND(StreamCaptureMode)>(createflag));
}

GCXX_FH auto StreamView::EndCapture() -> Graph {
  GraphView::deviceGraph_t pgraph{nullptr};
  GCXX_SAFE_RUNTIME_CALL(StreamEndCapture, "Failed to end Stream Capture",
                         this->getRawStream(), &pgraph);
  return Graph::CreateFromRaw(pgraph);
}

GCXX_FH auto StreamView::EndCaptureToGraph(const GraphView& graph = {})
  -> void {
  // When using BeginCaptureToGraph, the capture happens into the existing
  // graph, so the returned handle from EndCapture is the same as
  // graph.getRawGraph(). We just need to call EndCapture to finalize the
  // capture.
  GraphView::deviceGraph_t pgraph{nullptr};
  GCXX_SAFE_RUNTIME_CALL(StreamEndCapture, "Failed to end Stream Capture",
                         this->getRawStream(), &pgraph);
  // Assert that the returned graph is indeed the same as the one we passed in
  assert(pgraph == graph.getRawGraph() &&
         "EndCapture returned unexpected graph handle");
  (void)pgraph;  // Silence unused variable warning in release builds
}

#if GCXX_CUDA_MODE
GCXX_FH auto StreamView::IsCapturing() -> gcxx::flags::streamCaptureStatus {
  GCXX_RUNTIME_BACKEND(StreamCaptureStatus) status{};
  GCXX_SAFE_RUNTIME_CALL(StreamIsCapturing,
                         "Failed to query if the Stream is capturing", stream_,
                         &status);
  return flags::to_streamCaptureStatus(status);
}

GCXX_FH auto StreamView::GetCaptureInfo() -> CaptureInfo {
  GCXX_RUNTIME_BACKEND(StreamCaptureStatus) status{};
  unsigned long long id{};
  GraphView::deviceGraph_t graph{};
  const GraphView::deviceGraphNode_t* pDependencies = nullptr;
  std::size_t numdeps                               = 0;

  GCXX_SAFE_RUNTIME_CALL(StreamGetCaptureInfo,
                         "Failed to get Capture info of stream", stream_,
                         &status, &id, &graph, &pDependencies,
#if GCXX_CUDA_VERSION_GREATER_EQUAL(13, 0, 0)  // TODO : support dependency data
                         nullptr,
#endif

                         &numdeps);

  return {flags::to_streamCaptureStatus(status), id, GraphView(graph),
          pDependencies, numdeps};
}

GCXX_FH auto StreamView::UpdateCaptureDependencies(
  flags::StreamUpdateCaptureDependencies flag, deviceGraphNode_t* nodes,
  std::size_t numdeps) -> void {
  GCXX_SAFE_RUNTIME_CALL(StreamUpdateCaptureDependencies,
                         "Failed to update Dependencies to the cpatured graph",
                         stream_, nodes,
#if GCXX_CUDA_VERSION_GREATER_EQUAL(13, 0, 0)  // TODO : support dependency data
                         nullptr,
#endif
                         numdeps, static_cast<details_::flag_t>(flag));
}
#endif
GCXX_NAMESPACE_MAIN_END

#include <gcxx/macros/undefine_macros.hpp>

#endif