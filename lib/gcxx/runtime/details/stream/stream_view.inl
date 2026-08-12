// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_STREAM_STREAM_VIEW_INL_
#define GCXX_RUNTIME_DETAILS_STREAM_STREAM_VIEW_INL_

#include <gcxx/internal/prologue.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_FHC
StreamView::StreamView(deviceStream_t rawStream) GCXX_NOEXCEPT
    : m_stream(rawStream) {}

GCXX_FHC auto StreamView::getRawHandle() GCXX_CONST_NOEXCEPT -> deviceStream_t {
  return m_stream;
}

GCXX_FH auto StreamView::HasPendingWork() -> bool {
  const auto err = driver::streamQueryNothrow(m_stream);
  return !details_::nonFatalErrorQuery(err);
}

GCXX_FH auto StreamView::Synchronize() const -> void {
  driver::streamSynchronize(m_stream);
}

GCXX_FH auto StreamView::WaitOnEvent(const EventView& event,
                                     flags::eventWait waitFlag) const -> void {
  driver::StreamWaitEvent(this->m_stream, event.getRawHandle(),
                          static_cast<details_::flag_t>(waitFlag));
}

GCXX_NAMESPACE_MAIN_END()


#endif