#pragma once
#ifndef GCXX_RUNTIME_EVENT_EVENT_REF_HPP_
#define GCXX_RUNTIME_EVENT_EVENT_REF_HPP_

#include <chrono>
#include <utility>

#include <gpucxx/backend/backend.hpp>
#include <gpucxx/macros/define_macros.hpp>
#include <gpucxx/runtime/event/event_base.hpp>
#include <gpucxx/runtime/flags/eventflags.hpp>
#include <gpucxx/runtime/stream/stream_ref.hpp>


GCXX_DETAILS_BEGIN_NAMESPACE
/**
 * @brief Duration type representing milliseconds.
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

GCXX_DETAILS_END_NAMESPACE

GCXX_BEGIN_NAMESPACE

/**
 * @brief a non-owning wrapper for gpu events user is responsible for creating and destroying the event object
 * 
 */
class event_ref : public details_::event_ref {

 protected:
  using deviceEvent_t = GCXX_RUNTIME_BACKEND(Event_t);

 public:
  /// Default constructor
  event_ref() = default;

  /// Constructor from raw device event
  GCXX_CXPR event_ref(deviceEvent_t rawEvent) GCXX_NOEXCEPT
      : details_::event_ref(rawEvent) {}

  GCXX_CXPR event_ref(details_::event_ref eventRef) GCXX_NOEXCEPT
      : details_::event_ref(eventRef.get()) {}

  /// Delete constructor from `int`
  event_ref(int) = delete;

  /// Delete constructor from `nullptr`
  event_ref(std::nullptr_t) = delete;

  GCXX_FH auto HasOccurred() const -> bool;

  GCXX_FH auto Synchronize() const -> void;

  GCXX_FH auto RecordInStream(
    const stream_ref& stream      = details_::NULL_STREAM,
    flags::eventRecord recordFlag = flags::eventRecord::none) -> void;

  template <typename DurationT = details_::milliSec>
  GCXX_FH auto ElapsedTimeSince(const event_ref& startEvent) const
    -> DurationT;

  template <typename DurationT = details_::milliSec>
  GCXX_FH static auto ElapsedTimeBetween(const event_ref& startEvent,
                                           const event_ref& endEvent)
    -> DurationT {
    return endEvent.ElapsedTimeSince<DurationT>(startEvent);
  }
};

GCXX_END_NAMESPACE

#include <gpucxx/runtime/details/event_ref.inl>

#include <gpucxx/macros/undefine_macros.hpp>

#endif
