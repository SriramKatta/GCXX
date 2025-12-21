#pragma once
#ifndef GCXX_RUNTIME_STREAM_STREAM_VIEW_HPP_
#define GCXX_RUNTIME_STREAM_STREAM_VIEW_HPP_

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>
#include <gcxx/runtime/flags/stream_flags.hpp>


GCXX_NAMESPACE_MAIN_DETAILS_BEGIN
using deviceStream_t = GCXX_RUNTIME_BACKEND(Stream_t);
inline constexpr deviceStream_t NULL_STREAM{nullptr};  // NOLINT

// clang-format off
inline static const auto INVALID_STREAM = reinterpret_cast<deviceStream_t>(~0ULL);  // NOLINT
// clang-format on

GCXX_NAMESPACE_MAIN_DETAILS_END


GCXX_NAMESPACE_MAIN_BEGIN
class Event;
class GraphView;
class Graph;

class StreamView {
 protected:
  using deviceStream_t = details_::deviceStream_t;
  deviceStream_t stream_{details_::NULL_STREAM};  // NOLINT

 public:
  GCXX_FHC StreamView(deviceStream_t rawStream) GCXX_NOEXCEPT;

  StreamView()               = delete;
  StreamView(int)            = delete;
  StreamView(std::nullptr_t) = delete;

  GCXX_FH constexpr auto getRawStream() GCXX_CONST_NOEXCEPT->deviceStream_t;

  GCXX_FH constexpr operator deviceStream_t() GCXX_CONST_NOEXCEPT;

  GCXX_FH auto HasPendingWork() -> bool;

  GCXX_FH auto Synchronize() const -> void;

  GCXX_FH auto WaitOnEvent(
    const EventView& event,
    const flags::eventWait waitFlag = flags::eventWait::none) const -> void;

  GCXX_FH auto RecordEvent(
    const flags::eventCreate createflag = flags::eventCreate::none,
    const flags::eventRecord recordFlag = flags::eventRecord::none) const
    -> Event;

  GCXX_FH auto BeginCapture(const flags::streamCaptureMode createflag) -> void;

  GCXX_FH auto BeginCaptureToGraph(GraphView& graph_view,
                                   const flags::streamCaptureMode createflag)
    -> void;

  GCXX_FH auto EndCapture() -> Graph;

  /// @brief End stream capture and update the graph that was passed to
  /// BeginCaptureToGraph. Use this instead of EndCapture() when using
  /// BeginCaptureToGraph to avoid ownership issues.
  /// @param graph Reference to the same Graph passed to BeginCaptureToGraph
  GCXX_FH auto EndCaptureToGraph(GraphView& graph) -> void;
};

GCXX_NAMESPACE_MAIN_END


#include <gcxx/runtime/details/stream_view.inl>

#include <gcxx/macros/undefine_macros.hpp>
#endif