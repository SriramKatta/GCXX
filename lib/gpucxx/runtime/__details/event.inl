#pragma once
#ifndef GPUCXX_RUNTIME_DETAILS_EVENT_INL_
#define GPUCXX_RUNTIME_DETAILS_EVENT_INL_

#include <gpucxx/backend/backend.hpp>
#include <gpucxx/macros/define_macros.hpp>
#include <gpucxx/runtime/__flags/eventflags.hpp>
#include <gpucxx/runtime/runtime_error.hpp>

#include <utility>

GPUCXX_BEGIN_NAMESPACE

GPUCXX_FH Event::Event(const flags::eventCreate createFlag)
    : event_ref(details_::__invalid_event_) {
  GPUCXX_SAFE_RUNTIME_CALL(EventCreateWithFlags,
                           (&event_, static_cast<flag_t>(createFlag)));
}

GPUCXX_FH Event::~Event() {
  GPUCXX_SAFE_RUNTIME_CALL(EventDestroy, (event_));
}

GPUCXX_FH Event::Event(Event&& other) noexcept : event_ref(std::move(other)) {}

GPUCXX_FH auto Event::release() GPUCXX_NOEXCEPT -> event_ref {
  auto oldEvent = event_;
  event_        = details_::__invalid_event_;
  return event_ref(oldEvent);
}

// Implementation of recordEvent to break circular dependency
GPUCXX_FH auto stream_ref::recordEvent(const flags::eventCreate createflag,
                                       const flags::eventWait waitFlag) const
  -> Event {
  Event event(createflag);
  GPUCXX_SAFE_RUNTIME_CALL(EventRecord, (event.get(), this->get()));
  return event;
}

GPUCXX_END_NAMESPACE


#endif
