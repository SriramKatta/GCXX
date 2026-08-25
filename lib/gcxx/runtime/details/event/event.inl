// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_EVENT_EVENT_INL_
#define GCXX_RUNTIME_DETAILS_EVENT_EVENT_INL_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/flags/event_flags.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>

#include <utility>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_FH Event::Event(const flags::eventCreate createFlag)
    : EventView(driver::INVALID_EVENT) {
  m_event =
    driver::eventCreateWithFlags(static_cast<details_::flag_t>(createFlag));
}

GCXX_FH Event::~Event() {
  if (m_event != driver::INVALID_EVENT) {
    driver::eventDestroy(m_event);
    m_event = driver::INVALID_EVENT;
  }
}

GCXX_FH Event::Event(Event&& other) noexcept
    : EventView(std::exchange(other.m_event, driver::INVALID_EVENT)) {}

GCXX_FH auto Event::Release() GCXX_NOEXCEPT -> EventView {
  auto oldEvent = m_event;
  m_event       = driver::INVALID_EVENT;
  return {oldEvent};
}

GCXX_FH auto Event::operator=(Event&& other) noexcept -> Event& {
  if (this != &other)
    this->m_event = std::exchange(other.m_event, driver::INVALID_EVENT);
  return *this;
}

// Implementation of recordEvent to break circular dependency
GCXX_FH auto StreamView::recordEvent(const flags::eventCreate createflag,
                                     const flags::eventRecord recordFlag) const
  -> Event {
  Event event(createflag);
  event.recordInStream(*this, recordFlag);
  return event;
}

GCXX_NAMESPACE_MAIN_END()


#endif
