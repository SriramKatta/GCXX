#pragma once
#ifndef GCXX_RUNTIME_EVENT_EVENT_BASE_HPP_
#define GCXX_RUNTIME_EVENT_EVENT_BASE_HPP_

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>
#include <gcxx/runtime/flags/eventflags.hpp>


GCXX_NAMESPACE_MAIN_DETAILS_BEGIN
using deviceEvent_t = GCXX_RUNTIME_BACKEND(Event_t);
inline static GCXX_CXPR deviceEvent_t INVALID_EVENT{};  // Default null event

/**
 * @brief Base class for GPU events
 *
 * @details Serves as a base for all GPU event types, providing a common
 * interface.
 *
 */
class event_base {
 protected:
  using deviceEvent_t = details_::deviceEvent_t;
  deviceEvent_t event_{INVALID_EVENT};  // NOLINT

 public:
  /**
   * @brief Default construct a new event base object
   *
   */
  GCXX_CXPR event_base() = default;

  /**
   * @brief Construct a new event base object from raw device event
   *
   * @param device_event device event to be handled
   */
  GCXX_CXPR event_base(deviceEvent_t rawEvent) GCXX_NOEXCEPT
      : event_(rawEvent) {}

  // Disallow creation from `int`
  event_base(int) = delete;

  // Disallow creation from `nullptr`
  event_base(std::nullptr_t) = delete;

  GCXX_FHC auto get() GCXX_CONST_NOEXCEPT->deviceEvent_t { return event_; }

  GCXX_CXPR operator deviceEvent_t() GCXX_CONST_NOEXCEPT { return get(); }

  GCXX_CXPR explicit operator bool() GCXX_CONST_NOEXCEPT {
    return event_ != INVALID_EVENT;
  }

  GCXX_CXPR friend auto operator==(const event_base& lhs,
                                   const event_base& rhs) GCXX_NOEXCEPT->bool {
    return lhs.event_ == rhs.event_;
  }

  GCXX_CXPR friend auto operator!=(const event_base& lhs,
                                   const event_base& rhs) GCXX_NOEXCEPT->bool {
    return !(lhs == rhs);
  }
};

GCXX_NAMESPACE_MAIN_DETAILS_END

#include <gcxx/macros/undefine_macros.hpp>

#endif
