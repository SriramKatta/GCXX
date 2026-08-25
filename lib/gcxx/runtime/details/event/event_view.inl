// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_EVENT_EVENT_VIEW_INL_
#define GCXX_RUNTIME_DETAILS_EVENT_EVENT_VIEW_INL_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/flags/event_flags.hpp>

#include <gcxx/runtime/event/event_view.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>

#include <utility>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_CXPR
EventView::EventView(deviceEvent_t rawEvent) GCXX_NOEXCEPT : m_event(rawEvent) {
}

GCXX_CXPR
EventView::EventView(const EventView& eventRef) GCXX_NOEXCEPT
    : m_event(eventRef.getRawHandle()) {}

GCXX_FHC
auto EventView::getRawHandle() GCXX_CONST_NOEXCEPT -> deviceEvent_t {
  return m_event;
}

GCXX_CXPR EventView::operator bool() GCXX_CONST_NOEXCEPT {
  return m_event != driver::INVALID_EVENT;
}

GCXX_CXPR
auto EventView::operator=(const EventView& eventRef)
  GCXX_NOEXCEPT -> EventView& {
  m_event = eventRef.getRawHandle();
  return *this;
}

GCXX_CXPR
auto operator==(const EventView lhs, const EventView rhs) GCXX_NOEXCEPT->bool {
  return lhs.m_event == rhs.m_event;
}

GCXX_CXPR
auto operator!=(const EventView& lhs,
                const EventView& rhs) GCXX_NOEXCEPT->bool {
  return !(lhs == rhs);
}

GCXX_FH auto EventView::hasOccurred() const -> bool {
  const auto err = driver::eventQueryNoThrow(m_event);
  return details_::nonFatalErrorQuery(err);
}

GCXX_FH auto EventView::recordInStream(const flags::eventRecord recordFlag)
  -> void {
  recordInStream(StreamView::Null(), recordFlag);
}

GCXX_FH auto EventView::recordInStream(
  const StreamView& stream, const flags::eventRecord recordFlag) -> void {
  driver::eventRecordWithFlags(m_event, stream.getRawHandle(),
                               static_cast<details_::flag_t>(recordFlag));
}

GCXX_FH auto EventView::sync() const -> void {
  driver::eventSynchronize(m_event);
}

template <typename DurationT>
GCXX_FH auto EventView::elapsedTimeSince(const EventView& startEvent) const
  -> DurationT {
  this->sync();
  const auto ms =
    driver::eventElapsedTime(startEvent.getRawHandle(), getRawHandle());
  return ConvertDuration<DurationT>(ms);
}

template <typename DurationT>
GCXX_FH auto EventView::elapsedTimeBetween(
  const EventView& startEvent, const EventView& endEvent) -> DurationT {
  return endEvent.elapsedTimeSince<DurationT>(startEvent);
}

GCXX_NAMESPACE_MAIN_END()


#endif
