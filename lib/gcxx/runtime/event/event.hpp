#pragma once
#ifndef GCXX_RUNTIME_EVENT_EVENT_HPP_
#define GCXX_RUNTIME_EVENT_EVENT_HPP_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/event/event_view.hpp>
#include <gcxx/runtime/flags/event_flags.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

class Event : public EventView {
 private:
  GCXX_FH auto destroy() -> void;

 public:
  GCXX_FH Event(const flags::eventCreate createFlag = flags::eventCreate::None);

  GCXX_FH static auto Create(
    const flags::eventCreate createFlag = flags::eventCreate::None) -> Event;

  GCXX_FH ~Event();

  Event(const Event&) = delete;

  Event& operator=(const Event&) = delete;

  GCXX_FH Event(Event&& other) noexcept;

  GCXX_FH auto operator=(Event&& other) noexcept -> Event&;

  GCXX_FH auto Release() GCXX_NOEXCEPT->EventView;

  operator deviceEvent_t() = delete;
};

GCXX_NAMESPACE_MAIN_END

#include <gcxx/runtime/details/event/event.inl>

#endif
