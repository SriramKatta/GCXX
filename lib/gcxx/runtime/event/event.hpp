#pragma once
#ifndef GCXX_RUNTIME_EVENT_EVENT_HPP_
#define GCXX_RUNTIME_EVENT_EVENT_HPP_

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>
#include <gcxx/runtime/event/event_wrap.hpp>
#include <gcxx/runtime/flags/eventflags.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

class Event : public event_wrap {
 private:
  GCXX_FH auto destroy() -> void;

 public:
  GCXX_FH Event(const flags::eventCreate createFlag = flags::eventCreate::none);

  GCXX_FH static auto Create(
    const flags::eventCreate createFlag = flags::eventCreate::none) -> Event {
    return {createFlag};
  };

  GCXX_FH ~Event();

  Event(const Event&) = delete;

  Event& operator=(const Event&) = delete;

  GCXX_FH Event(Event&& other) noexcept
      : event_wrap(std::exchange(other.event_, details_::INVALID_EVENT)) {}

  GCXX_FH auto operator=(Event&& other) noexcept -> Event& {
    if (this != &other)
      this->event_ = std::exchange(other.event_, details_::INVALID_EVENT);
    return *this;
  }

  GCXX_FH auto release() GCXX_NOEXCEPT -> event_wrap;

  operator deviceEvent_t() = delete;
};

GCXX_NAMESPACE_MAIN_END

#include <gcxx/runtime/details/event.inl>

#endif
