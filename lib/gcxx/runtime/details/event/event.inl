#pragma once
#ifndef GCXX_RUNTIME_DETAILS_EVENT_EVENT_INL_
#define GCXX_RUNTIME_DETAILS_EVENT_EVENT_INL_

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>
#include <gcxx/runtime/flags/event_flags.hpp>
#include <gcxx/runtime/runtime_error.hpp>

#include <utility>

GCXX_NAMESPACE_MAIN_BEGIN

GCXX_FH auto Event::Create(const flags::eventCreate createFlag) -> Event {
  return {createFlag};
};

GCXX_FH Event::Event(const flags::eventCreate createFlag)
    : EventView(details_::INVALID_EVENT) {
  GCXX_SAFE_RUNTIME_CALL(EventCreateWithFlags, "Failed to create GPU Event",
                         &event_, static_cast<details_::flag_t>(createFlag));
}

GCXX_FH Event::~Event() {
  if (event_ != details_::INVALID_EVENT) {
    GCXX_SAFE_RUNTIME_CALL(EventDestroy, "Failed to destroy GPU Event", event_);
  }
}

GCXX_FH Event::Event(Event&& other) noexcept
    : EventView(std::exchange(other.event_, details_::INVALID_EVENT)) {}

GCXX_FH auto Event::Release() GCXX_NOEXCEPT->EventView {
  auto oldEvent = event_;
  event_        = details_::INVALID_EVENT;
  return {oldEvent};
}

GCXX_FH auto Event::operator=(Event&& other) noexcept -> Event& {
  if (this != &other)
    this->event_ = std::exchange(other.event_, details_::INVALID_EVENT);
  return *this;
}

// Implementation of recordEvent to break circular dependency
GCXX_FH auto StreamView::RecordEvent(const flags::eventCreate createflag,
                                     const flags::eventRecord recordFlag) const
  -> Event {
  Event event(createflag);
  event.RecordInStream(this->getRawStream(), recordFlag);
  return event;
}

GCXX_NAMESPACE_MAIN_END


#endif
