// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_STREAM_STREAM_VIEW_INL_
#define GCXX_RUNTIME_DETAILS_STREAM_STREAM_VIEW_INL_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/event/event_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_FHC
StreamView::StreamView(deviceStream_t rawStream) GCXX_NOEXCEPT
    : m_stream(rawStream) {}

GCXX_FHC auto StreamView::getRawHandle() GCXX_CONST_NOEXCEPT -> deviceStream_t {
  return m_stream;
}

GCXX_FH auto StreamView::hasPendingWork() -> bool {
  const auto err = driver::streamQueryNothrow(m_stream);
  return !details_::nonFatalErrorQuery(err);
}

GCXX_FH auto StreamView::sync() const -> void {
  driver::streamSynchronize(m_stream);
}

GCXX_FH auto StreamView::waitOnEvent(const EventView& event,
                                     flags::eventWait waitFlag) const -> void {
  driver::StreamWaitEvent(this->m_stream, event.getRawHandle(),
                          static_cast<details_::flag_t>(waitFlag));
}

// The Event<->Stream cross-methods live here rather than in event_view.inl:
// each side's .inl needs the OTHER class complete, so keeping them together
// in one of the two breaks the mutual tail-include cycle. Any TU entering
// through either header reaches this file with both classes defined.
GCXX_FH auto EventView::recordInStream(const flags::eventRecord recordFlag)
  -> void {
  recordInStream(StreamView::Null(), recordFlag);
}

GCXX_FH auto EventView::recordInStream(
  const StreamView& stream, const flags::eventRecord recordFlag) -> void {
  driver::eventRecordWithFlags(m_event, stream.getRawHandle(),
                               static_cast<details_::flag_t>(recordFlag));
}

GCXX_NAMESPACE_MAIN_END()


#endif