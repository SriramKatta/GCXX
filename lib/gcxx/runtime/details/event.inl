#pragma once
#ifndef GCXX_RUNTIME_DETAILS_EVENT_INL_
#define GCXX_RUNTIME_DETAILS_EVENT_INL_

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>
#include <gcxx/runtime/flags/eventflags.hpp>
#include <gcxx/runtime/runtime_error.hpp>

#include <utility>

GCXX_NAMESPACE_MAIN_BEGIN

GCXX_FH Event::Event(const flags::eventCreate createFlag)
    : event_wrap(details_::INVALID_EVENT) {
  GCXX_SAFE_RUNTIME_CALL(EventCreateWithFlags, "Failed to create GPU Event",
                         &event_, static_cast<flag_t>(createFlag));
}

GCXX_FH Event::~Event() {
  if (event_ != details_::INVALID_EVENT) {
    GCXX_SAFE_RUNTIME_CALL(EventDestroy, "Failed to destroy GPU Event", event_);
  }
}

GCXX_FH auto Event::release() GCXX_NOEXCEPT->event_wrap {
  auto oldEvent = event_;
  event_        = details_::INVALID_EVENT;
  return {oldEvent};
}

// Implementation of recordEvent to break circular dependency
GCXX_FH auto stream_wrap::recordEvent(const flags::eventCreate createflag,
                                      const flags::eventRecord recordFlag) const
  -> Event {
  Event event(createflag);
  event.RecordInStream(this->get(), recordFlag);
  return event;
}

GCXX_NAMESPACE_MAIN_END


#endif
