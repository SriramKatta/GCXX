// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_STREAM_STREAM_VIEW_HPP_
#define GCXX_RUNTIME_STREAM_STREAM_VIEW_HPP_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/flags/stream_flags.hpp>
#include <gcxx/runtime_backend/backend_graph.hpp>
#include <gcxx/runtime_backend/backend_stream.hpp>

GCXX_NAMESPACE_MAIN_BEGIN
class Event;
class GraphView;
class Graph;
struct CaptureInfo;

class StreamView {
 private:
  using deviceGraphNode_t = driver::deviceGraphNode_t;

 protected:
  using deviceStream_t = driver::deviceStream_t;
  deviceStream_t stream_{driver::NULL_STREAM};  // NOLINT

 public:
  GCXX_FHC StreamView(deviceStream_t rawStream) GCXX_NOEXCEPT;

  StreamView()               = delete;
  StreamView(int)            = delete;
  StreamView(std::nullptr_t) = delete;

  static StreamView& Null() {
    static StreamView s(driver::NULL_STREAM);
    return s;
  }

  GCXX_FH constexpr auto getRawStream() GCXX_CONST_NOEXCEPT -> deviceStream_t;

  GCXX_FH constexpr operator deviceStream_t() GCXX_CONST_NOEXCEPT;

  GCXX_FH auto HasPendingWork() -> bool;

  GCXX_FH auto Synchronize() const -> void;

  GCXX_FH auto WaitOnEvent(
    const EventView& event,
    const flags::eventWait waitFlag = flags::eventWait::None) const -> void;

  GCXX_FH auto RecordEvent(
    const flags::eventCreate createflag = flags::eventCreate::None,
    const flags::eventRecord recordFlag = flags::eventRecord::None) const
    -> Event;

  GCXX_FHDC auto isNullStream() -> bool {
    return stream_ == driver::NULL_STREAM;
  }

  GCXX_FHD auto isInvalidStream() -> bool {
    return stream_ == driver::INVALID_STREAM;
  }

  GCXX_FH auto BeginCapture(const flags::streamCaptureMode createflag) -> void;

  GCXX_FH auto BeginCaptureToGraph(
    GraphView& graph_view, const flags::streamCaptureMode createflag) -> void;

  GCXX_FH auto EndCapture() -> Graph;

  /// @brief End stream capture and update the graph that was passed to
  /// BeginCaptureToGraph. Use this instead of EndCapture() when using
  /// BeginCaptureToGraph to avoid ownership issues.
  /// @param graph Reference to the same Graph passed to BeginCaptureToGraph
  GCXX_FH auto EndCaptureToGraph(const GraphView& graph) -> void;

#if GCXX_CUDA_MODE
  GCXX_FH auto IsCapturing() -> gcxx::flags::streamCaptureStatus;

  GCXX_FH auto GetCaptureInfo() -> CaptureInfo;

  GCXX_FH auto UpdateCaptureDependencies(
    flags::StreamUpdateCaptureDependencies flag, deviceGraphNode_t* nodes,
    std::size_t numdeps) -> void;
#endif
};

GCXX_NAMESPACE_MAIN_END


#include <gcxx/runtime/details/stream/stream_view.inl>

#include <gcxx/macros/undefine_macros.hpp>
#endif