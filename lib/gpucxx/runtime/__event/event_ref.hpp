#pragma once
#ifndef GPUCXX_RUNTIME_EVENT_EVENT_REF_HPP_
#define GPUCXX_RUNTIME_EVENT_EVENT_REF_HPP_

#include <chrono>
#include <utility>

#include <gpucxx/backend/backend.hpp>
#include <gpucxx/macros/define_macros.hpp>
#include <gpucxx/runtime/__flags/eventflags.hpp>
#include <gpucxx/runtime/__stream/stream_ref.hpp>


GPUCXX_DETAILS_BEGIN_NAMESPACE
/**
 * @brief Duration type representing milliseconds.
 *
 */
using nanoSecDuration_t  = std::chrono::duration<float, std::nano>;
using microSecDuration_t = std::chrono::duration<float, std::micro>;
using milliSecDuration_t = std::chrono::duration<float, std::milli>;
using secDuration_t      = std::chrono::duration<float>;

template <typename DurationT>
inline auto ConvertDuration(float ms) -> DurationT {
  return std::chrono::duration_cast<DurationT>(milliSecDuration_t(ms));
}

GPUCXX_DETAILS_END_NAMESPACE

GPUCXX_BEGIN_NAMESPACE

/**
 * @brief a non-owning wrapper for gpu events user is responsible for creating and destroying the event object
 * 
 */
class event_ref : public details_::event_ref {

 protected:
  using deviceEvent_t = GPUCXX_RUNTIME_BACKEND(Event_t);

 public:
  /// Default constructor
  event_ref() = delete;

  /// Constructor from raw device event
  GPUCXX_CXPR event_ref(deviceEvent_t __evt) GPUCXX_NOEXCEPT
      : details_::event_ref(__evt) {}

  GPUCXX_CXPR event_ref(details_::event_ref __evt) GPUCXX_NOEXCEPT
      : details_::event_ref(__evt.get()) {}

  /// Delete copy constructor
  event_ref(const event_ref&) = delete;

  /// Delete copy assignment operator
  event_ref& operator=(const event_ref&) = delete;

  /// Default move constructor
  GPUCXX_CXPR event_ref(event_ref&& other) GPUCXX_NOEXCEPT
      : details_::event_ref(other.event_) {
    other.event_ = __invalid_event_;
  }

  /// Default move assignment operator
  event_ref& operator=(event_ref&& other) GPUCXX_NOEXCEPT {
    if (this != &other) {
      this->event_ = other.event_;
      other.event_ = __invalid_event_;
    }
    return *this;
  }

  /// Delete constructor from `int`
  event_ref(int) = delete;

  /// Delete constructor from `nullptr`
  event_ref(std::nullptr_t) = delete;


  GPUCXX_FH auto HasOccurred() const -> bool;

  GPUCXX_FH auto Synchronize() const -> void;

  GPUCXX_FH auto RecordInStream(
    const stream_ref& stream      = details_::__null_stream_,
    flags::eventRecord recordFlag = flags::eventRecord::none) -> void;

  template <typename DurationT = details_::milliSecDuration_t>
  GPUCXX_FH auto ElapsedTimeSince(const event_ref& startEvent) const
    -> DurationT;

  template <typename DurationT = details_::milliSecDuration_t>
  GPUCXX_FH static auto ElapsedTimeBetween(const event_ref& startEvent,
                                           const event_ref& endEvent)
    -> DurationT {
    return endEvent.ElapsedTimeSince<DurationT>(startEvent);
  }
};

GPUCXX_END_NAMESPACE

#include <gpucxx/runtime/__details/event_ref.inl>

#include <gpucxx/macros/undefine_macros.hpp>

#endif
