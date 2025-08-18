#pragma once
#ifndef GPUCXX_RUNTIME_EVENT_EVENT_BASE_HPP_
#define GPUCXX_RUNTIME_EVENT_EVENT_BASE_HPP_

#include <gpucxx/backend/backend.hpp>
#include <gpucxx/macros/define_macros.hpp>
#include <gpucxx/runtime/__flags/eventflags.hpp>


GPUCXX_BEGIN_NAMESPACE

class event_base {

 protected:
  using deviceEvent_t = GPUCXX_RUNTIME_BACKEND(Event_t);

 public:
  constexpr event_base(deviceEvent_t __evt) noexcept : event_(__evt) {}

  event_base()               = delete;
  event_base(int)            = delete;
  event_base(std::nullptr_t) = delete;

  GPUCXX_FHD constexpr auto get() const noexcept -> deviceEvent_t {
    return event_;
  }
 protected:
  deviceEvent_t event_{static_cast<deviceEvent_t>(0ULL)};
};

GPUCXX_END_NAMESPACE

#include <gpucxx/macros/undefine_macros.hpp>

#endif
