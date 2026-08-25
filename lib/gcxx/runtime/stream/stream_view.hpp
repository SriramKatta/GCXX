// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_STREAM_STREAM_VIEW_HPP_
#define GCXX_RUNTIME_STREAM_STREAM_VIEW_HPP_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/macros/template_helper_macros.hpp>
#include <gcxx/runtime/flags/event_flags.hpp>
#include <gcxx/runtime/flags/memory_flags.hpp>
#include <gcxx/runtime/flags/stream_flags.hpp>
#include <gcxx/runtime/memory/spans/spans.hpp>
#include <gcxx/runtime_backend/backend_handles.hpp>
#include <gcxx/runtime_backend/backend_stream.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()
class Event;
class EventView;
class GraphView;
class Graph;
struct CaptureInfo;

class StreamView {

 public:
  using deviceGraphNode_t = driver::deviceGraphNode_t;
  using deviceStream_t    = driver::deviceStream_t;
  using raw_handle_type   = driver::deviceStream_t;

  explicit GCXX_FHC StreamView(deviceStream_t rawStream) GCXX_NOEXCEPT;

  StreamView()               = delete;
  StreamView(int)            = delete;
  StreamView(std::nullptr_t) = delete;

  static StreamView& Null() {
    static StreamView s(driver::NULL_STREAM);
    return s;
  }

  GCXX_FH constexpr auto getRawHandle() GCXX_CONST_NOEXCEPT -> deviceStream_t;

  GCXX_FH auto hasPendingWork() -> bool;

  GCXX_FH auto sync() const -> void;

  GCXX_FH auto waitOnEvent(
    const EventView& event,
    flags::eventWait waitFlag = flags::eventWait::None) const -> void;

  GCXX_FH auto recordEvent(
    flags::eventCreate createflag = flags::eventCreate::None,
    flags::eventRecord recordFlag = flags::eventRecord::None) const -> Event;

  // TODO: No op in HIP.
  GCXX_TEMPLATE(typename Span)
  GCXX_REQUIRES(is_span_like_v<Span>)
  GCXX_FH auto attachMemAsync(
    Span&& mem,
    flags::memAttach flag = flags::memAttach::Single) const -> void {
    driver::streamAttachMemAsync(
      m_stream, static_cast<void*>(details_::to_address(details_::data(mem))),
      details_::size(mem) * sizeof(span_element_t<Span>),
      static_cast<details_::flag_t>(flag));
  }

  GCXX_FHDC auto isNullStream() const -> bool {
    return m_stream == driver::NULL_STREAM;
  }

  GCXX_FHD auto isInvalidStream() const -> bool {
    return m_stream == driver::INVALID_STREAM;
  }

  GCXX_FH auto beginCapture(flags::streamCaptureMode createflag) const -> void;

  GCXX_FH auto beginCaptureToGraph(
    GraphView& graph_view, flags::streamCaptureMode createflag) const -> void;

  GCXX_FH auto endCapture() const -> Graph;

  // Updates the graph passed to beginCaptureToGraph; avoids ownership issues.
  GCXX_FH auto endCaptureToGraph(const GraphView& graph) const -> void;

#if GCXX_CUDA_MODE()
  GCXX_FH auto isCapturing() const -> gcxx::flags::streamCaptureStatus;

  GCXX_FH auto getCaptureInfo() const -> CaptureInfo;

  GCXX_FH auto updateCaptureDependencies(
    flags::StreamUpdateCaptureDependencies flag, deviceGraphNode_t* nodes,
    std::size_t numdeps) const -> void;
#endif

 protected:
  deviceStream_t m_stream{driver::NULL_STREAM};  // NOLINT
};

GCXX_NAMESPACE_MAIN_END()


#include <gcxx/runtime/details/stream/stream_view.inl>


#endif