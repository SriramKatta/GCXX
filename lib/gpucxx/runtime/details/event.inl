#pragma once
#ifndef GPUCXX_RUNTIME_DETAILS_EVENT_INL_
#define GPUCXX_RUNTIME_DETAILS_EVENT_INL_

#include <gpucxx/backend/backend.hpp>
#include <gpucxx/macros/define_macros.hpp>
#include <gpucxx/runtime/flags/eventflags.hpp>
#include <gpucxx/runtime/runtime_error.hpp>

#include <utility>

GPUCXX_BEGIN_NAMESPACE

GPUCXX_FH Event::Event(const flags::eventCreate createFlag)
    : event_ref(details_::INVALID_EVENT) {
  GPUCXX_SAFE_RUNTIME_CALL(EventCreateWithFlags,
                           (&event_, static_cast<flag_t>(createFlag)));
}

GPUCXX_FH Event::~Event() {
  if (event_ != details_::INVALID_EVENT) {
    GPUCXX_SAFE_RUNTIME_CALL(EventDestroy, (event_));
  }
}

GPUCXX_FH auto Event::release() GPUCXX_NOEXCEPT -> event_ref {
  auto oldEvent = event_;
  event_        = details_::INVALID_EVENT;
  return {oldEvent};
}

// Implementation of recordEvent to break circular dependency
GPUCXX_FH auto stream_ref::recordEvent(
  const flags::eventCreate createflag,
  const flags::eventRecord recordFlag) const -> Event {
  Event event(createflag);
  event.RecordInStream(this->get(), recordFlag);
  return event;
}

GPUCXX_END_NAMESPACE


#endif
