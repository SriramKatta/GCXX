#pragma once
#ifndef GCXX_RUNTIME_EVENT_EVENT_VIEW_HPP_
#define GCXX_RUNTIME_EVENT_EVENT_VIEW_HPP_

#include <chrono>
#include <utility>

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>
#include <gcxx/runtime/flags/eventflags.hpp>

GCXX_NAMESPACE_MAIN_DETAILS_BEGIN

using deviceEvent_t = GCXX_RUNTIME_BACKEND(Event_t);
inline static GCXX_CXPR deviceEvent_t INVALID_EVENT{};  // Default null event

GCXX_NAMESPACE_MAIN_DETAILS_END


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

class StreamView;

/**
 * @brief a non-owning wrapper for gpu events user is responsible for creating
 * and destroying the event object
 *
 */
class EventView {
 protected:
  using deviceEvent_t = GCXX_RUNTIME_BACKEND(Event_t);
  deviceEvent_t event_{details_::INVALID_EVENT};  // NOLINT

 public:
  /// Default constructor
  EventView() = default;

  /// Constructor from raw device event
  GCXX_CXPR EventView(deviceEvent_t rawEvent) GCXX_NOEXCEPT;

  GCXX_CXPR EventView(const EventView& eventRef) GCXX_NOEXCEPT;

  GCXX_FHC auto getRawEvent() GCXX_CONST_NOEXCEPT->deviceEvent_t;

  GCXX_CXPR operator deviceEvent_t() GCXX_CONST_NOEXCEPT;

  GCXX_CXPR explicit operator bool() GCXX_CONST_NOEXCEPT;

  GCXX_CXPR friend auto operator==(const EventView lhs,
                                   const EventView rhs) GCXX_NOEXCEPT->bool;

  GCXX_CXPR friend auto operator!=(const EventView& lhs,
                                   const EventView& rhs) GCXX_NOEXCEPT->bool;

  /// Delete constructor from `int`
  EventView(int) = delete;

  /// Delete constructor from `nullptr`
  EventView(std::nullptr_t) = delete;

  GCXX_FH auto HasOccurred() const -> bool;

  GCXX_FH auto Synchronize() const -> void;

  GCXX_FH auto RecordInStream(
    flags::eventRecord recordFlag = flags::eventRecord::none) -> void;

  GCXX_FH auto RecordInStream(
    const StreamView& stream,
    flags::eventRecord recordFlag = flags::eventRecord::none) -> void;

  template <typename DurationT = milliSec>
  GCXX_FH auto ElapsedTimeSince(const EventView& startEvent) const -> DurationT;

  template <typename DurationT = milliSec>
  GCXX_FH static auto ElapsedTimeBetween(const EventView& startEvent,
                                         const EventView& endEvent)
    -> DurationT;
};

GCXX_NAMESPACE_MAIN_END

#include <gcxx/runtime/details/event_view.inl>

#include <gcxx/macros/undefine_macros.hpp>

#endif
