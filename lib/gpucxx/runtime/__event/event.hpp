#pragma once
#ifndef GPUCXX_RUNTIME_EVENT_EVENT_HPP_
#define GPUCXX_RUNTIME_EVENT_EVENT_HPP_

#include <gpucxx/backend/backend.hpp>
#include <gpucxx/macros/define_macros.hpp>
#include <gpucxx/runtime/__event/event_ref.hpp>
#include <gpucxx/runtime/__flags/eventflags.hpp>

GPUCXX_BEGIN_NAMESPACE

class Event : public event_ref {
 private:
  GPUCXX_FH auto destroy() -> void;

 public:
  GPUCXX_FH Event(
    const flags::eventCreate createFlag = flags::eventCreate::none);

  GPUCXX_FH static auto Create(
    const flags::eventCreate createFlag = flags::eventCreate::none) -> Event {
    return Event(createFlag);
  };

  GPUCXX_FH ~Event();

  Event(const Event&) = delete;

  Event& operator=(const Event&) = delete;

  GPUCXX_FH Event(Event&& other) noexcept;

  GPUCXX_FH auto release() GPUCXX_NOEXCEPT -> event_ref;

  operator deviceEvent_t() = delete;
};

GPUCXX_END_NAMESPACE

#include <gpucxx/runtime/__details/event.inl>


#endif
