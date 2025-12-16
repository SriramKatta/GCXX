#pragma once
#ifndef GCXX_RUNTIME_EVENT_EVENT_VIEW_HPP_
#define GCXX_RUNTIME_EVENT_EVENT_VIEW_HPP_

#include <chrono>
#include <utility>

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>
#include <gcxx/runtime/event/event_base.hpp>
#include <gcxx/runtime/flags/eventflags.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>


GCXX_NAMESPACE_MAIN_BEGIN

/**
 * @brief Duration type aliases for time measurements.
 *
 */
using nanoSec  = std::chrono::duration<float, std::nano>;
using microSec = std::chrono::duration<float, std::micro>;
using milliSec = std::chrono::duration<float, std::milli>;
using sec      = std::chrono::duration<float>;

template <typename DurationT>
inline auto ConvertDuration(float ms) -> DurationT {
  return std::chrono::duration_cast<DurationT>(milliSec(ms));
}

/**
 * @brief a non-owning wrapper for gpu events user is responsible for creating
 * and destroying the event object
 *
 */
class EventView : public details_::event_base {
 public:
  /// Default constructor
  EventView() = default;

  /// Constructor from raw device event
  GCXX_CXPR EventView(deviceEvent_t rawEvent) GCXX_NOEXCEPT
      : details_::event_base(rawEvent) {}

  GCXX_CXPR EventView(details_::event_base eventRef) GCXX_NOEXCEPT
      : details_::event_base(eventRef.get()) {}

  /// Delete constructor from `int`
  EventView(int) = delete;

  /// Delete constructor from `nullptr`
  EventView(std::nullptr_t) = delete;

  GCXX_FH auto HasOccurred() const -> bool;

  GCXX_FH auto Synchronize() const -> void;

  GCXX_FH auto RecordInStream(
    const StreamView& stream      = details_::NULL_STREAM,
    flags::eventRecord recordFlag = flags::eventRecord::none) -> void;

  template <typename DurationT = milliSec>
  GCXX_FH auto ElapsedTimeSince(const EventView& startEvent) const -> DurationT;

  template <typename DurationT = milliSec>
  GCXX_FH static auto ElapsedTimeBetween(const EventView& startEvent,
                                         const EventView& endEvent)
    -> DurationT {
    return endEvent.ElapsedTimeSince<DurationT>(startEvent);
  }
};

GCXX_NAMESPACE_MAIN_END

#include <gcxx/runtime/details/event_view.inl>

#include <gcxx/macros/undefine_macros.hpp>

#endif
