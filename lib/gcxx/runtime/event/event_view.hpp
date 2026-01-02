#pragma once
#ifndef GCXX_RUNTIME_EVENT_EVENT_VIEW_HPP_
#define GCXX_RUNTIME_EVENT_EVENT_VIEW_HPP_

#include <chrono>
#include <utility>

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>
#include <gcxx/runtime/flags/event_flags.hpp>

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

/**
 * @brief Converts a time value in milliseconds to the specified duration type.
 *
 * @tparam DurationT The target duration type (e.g., nanoSec, microSec,
 * milliSec, sec)
 * @param ms Time value in milliseconds to convert
 * @return The converted duration in the specified type
 */
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
  /** @brief Default constructor - creates an EventView with an invalid/null
   * event */
  EventView() = default;


  /** @brief Constructor from raw device event - wraps an existing GPU event
   * handle */
  GCXX_CXPR EventView(deviceEvent_t rawEvent) GCXX_NOEXCEPT;

  /** @brief Copy constructor - creates a shallow copy sharing the same
   * underlying event */
  GCXX_CXPR EventView(const EventView& eventRef) GCXX_NOEXCEPT;

  GCXX_CXPR auto operator=(const EventView& eventRef) GCXX_NOEXCEPT ->EventView&;

  /** @brief Returns the underlying raw GPU event handle */
  GCXX_FHC auto getRawEvent() GCXX_CONST_NOEXCEPT->deviceEvent_t;

  /** @brief Implicit conversion operator to raw device event type */
  GCXX_CXPR operator deviceEvent_t() GCXX_CONST_NOEXCEPT;

  /** @brief Boolean conversion - returns true if the event is valid (not
   * null/invalid) */
  GCXX_CXPR explicit operator bool() GCXX_CONST_NOEXCEPT;

  /** @brief Equality comparison - checks if two EventViews reference the same
   * event */
  GCXX_CXPR friend auto operator==(const EventView lhs,
                                   const EventView rhs) GCXX_NOEXCEPT->bool;

  /** @brief Inequality comparison - checks if two EventViews reference
   * different events */
  GCXX_CXPR friend auto operator!=(const EventView& lhs,
                                   const EventView& rhs) GCXX_NOEXCEPT->bool;

  /** @brief Deleted constructor from int - prevents accidental implicit
   * conversions */
  EventView(int) = delete;

  /** @brief Deleted constructor from nullptr - prevents null initialization
   * ambiguity */
  EventView(std::nullptr_t) = delete;

  /**
   * @brief Queries whether the event has been recorded and all preceding work
   * completed
   * @return true if the event has occurred, false if still pending
   */
  GCXX_FH auto HasOccurred() const -> bool;

  /** @brief Blocks the calling CPU thread until the event has been recorded and
   * completed */
  GCXX_FH auto Synchronize() const -> void;

  /**
   * @brief Records the event in the default/null stream with optional recording
   * flags
   * @param recordFlag Optional flags to control event recording behavior
   */
  GCXX_FH auto RecordInStream(
    flags::eventRecord recordFlag = flags::eventRecord::None) -> void;

  /**
   * @brief Records the event in the specified stream with optional recording
   * flags
   * @param stream The stream in which to record this event
   * @param recordFlag Optional flags to control event recording behavior
   */
  GCXX_FH auto RecordInStream(
    const StreamView& stream,
    flags::eventRecord recordFlag = flags::eventRecord::None) -> void;

  /**
   * @brief Computes the elapsed time from startEvent to this event
   *
   * Both events must have been recorded before calling this method
   * @tparam DurationT The duration type for the result (default: milliseconds)
   * @param startEvent The earlier event to measure from
   * @return Elapsed time between startEvent and this event
   */
  template <typename DurationT = milliSec>
  GCXX_FH auto ElapsedTimeSince(const EventView& startEvent) const -> DurationT;

  /**
   * @brief Computes the elapsed time between two events (static version)
   *
   * Both events must have been recorded before calling this method
   * @tparam DurationT The duration type for the result (default: milliseconds)
   * @param startEvent The earlier event
   * @param endEvent The later event
   * @return Elapsed time between startEvent and endEvent
   */
  template <typename DurationT = milliSec>
  GCXX_FH static auto ElapsedTimeBetween(const EventView& startEvent,
                                         const EventView& endEvent)
    -> DurationT;
};

GCXX_NAMESPACE_MAIN_END

#include <gcxx/runtime/details/event/event_view.inl>

#include <gcxx/macros/undefine_macros.hpp>

#endif
