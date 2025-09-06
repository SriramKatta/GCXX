#pragma once
#ifndef GPUCXX_RUNTIME_EVENT_EVENT_BASE_HPP_
#define GPUCXX_RUNTIME_EVENT_EVENT_BASE_HPP_

#include <gpucxx/backend/backend.hpp>
#include <gpucxx/macros/define_macros.hpp>
#include <gpucxx/runtime/__flags/eventflags.hpp>


GPUCXX_DETAILS_BEGIN_NAMESPACE
// clang-format off
  using deviceEvent_t = GPUCXX_RUNTIME_BACKEND(Event_t);
  inline static GPUCXX_CXPR deviceEvent_t __invalid_event_{};  // Default null event
// clang-format on

/**
 * @brief Base class for GPU events
 *
 * @details Serves as a base for all GPU event types, providing a common interface.
 *
 */
class event_ref {
 protected:
  using deviceEvent_t = GPUCXX_RUNTIME_BACKEND(Event_t);

 public:
  /**
   * @brief Default construct a new event base object
   *
   */
  event_ref() = default;

  /**
  * @brief Construct a new event base object from raw device event
  *
  * @param device_event device event to be handled
  */
  GPUCXX_CXPR event_ref(deviceEvent_t __evt) GPUCXX_NOEXCEPT : event_(__evt) {}

  /// Disallow creation from `int`
  event_ref(int) = delete;

  /// Disallow creation from `nullptr`
  event_ref(std::nullptr_t) = delete;

  GPUCXX_FHC auto get() GPUCXX_CONST_NOEXCEPT -> deviceEvent_t {
    return event_;
  }

  GPUCXX_CXPR operator deviceEvent_t() GPUCXX_CONST_NOEXCEPT { return get(); }

  GPUCXX_CXPR explicit operator bool() GPUCXX_CONST_NOEXCEPT {
    return event_ != __invalid_event_;
  }

  GPUCXX_CXPR friend auto operator==(const event_ref& lhs, const event_ref& rhs)
    GPUCXX_NOEXCEPT->bool {
    return lhs.event_ == rhs.event_;
  }

  GPUCXX_CXPR friend auto operator!=(const event_ref& lhs, const event_ref& rhs)
    GPUCXX_NOEXCEPT->bool {
    return !(lhs == rhs);
  }


 protected:
  deviceEvent_t event_{__invalid_event_};
};

GPUCXX_DETAILS_END_NAMESPACE

#include <gpucxx/macros/undefine_macros.hpp>

#endif
