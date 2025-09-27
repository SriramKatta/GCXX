#pragma once
#ifndef GCXX_RUNTIME_EVENT_EVENT_BASE_HPP_
#define GCXX_RUNTIME_EVENT_EVENT_BASE_HPP_

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>
#include <gcxx/runtime/flags/eventflags.hpp>


GCXX_DETAILS_BEGIN_NAMESPACE
// clang-format off
  using deviceEvent_t = GCXX_RUNTIME_BACKEND(Event_t);
  inline static GCXX_CXPR deviceEvent_t INVALID_EVENT{};  // Default null event
// clang-format on

/**
 * @brief Base class for GPU events
 *
 * @details Serves as a base for all GPU event types, providing a common interface.
 *
 */
class event_ref {
 protected:
  deviceEvent_t event_{INVALID_EVENT};  // NOLINT

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
  GCXX_CXPR event_ref(deviceEvent_t rawEvent) GCXX_NOEXCEPT
      : event_(rawEvent) {}

  /// Disallow creation from `int`
  event_ref(int) = delete;

  /// Disallow creation from `nullptr`
  event_ref(std::nullptr_t) = delete;

  GCXX_FHC auto get() GCXX_CONST_NOEXCEPT -> deviceEvent_t {
    return event_;
  }

  GCXX_CXPR operator deviceEvent_t() GCXX_CONST_NOEXCEPT { return get(); }

  GCXX_CXPR explicit operator bool() GCXX_CONST_NOEXCEPT {
    return event_ != INVALID_EVENT;
  }

  GCXX_CXPR friend auto operator==(const event_ref& lhs, const event_ref& rhs)
    GCXX_NOEXCEPT->bool {
    return lhs.event_ == rhs.event_;
  }

  GCXX_CXPR friend auto operator!=(const event_ref& lhs, const event_ref& rhs)
    GCXX_NOEXCEPT->bool {
    return !(lhs == rhs);
  }
};

GCXX_DETAILS_END_NAMESPACE

#include <gcxx/macros/undefine_macros.hpp>

#endif
