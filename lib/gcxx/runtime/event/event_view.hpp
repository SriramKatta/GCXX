// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_EVENT_EVENT_VIEW_HPP_
#define GCXX_RUNTIME_EVENT_EVENT_VIEW_HPP_

#include <chrono>
#include <utility>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/flags/event_flags.hpp>
#include <gcxx/runtime_backend/backend_event.hpp>


GCXX_NAMESPACE_MAIN_BEGIN()

using nanoSec  = std::chrono::duration<float, std::nano>;
using microSec = std::chrono::duration<float, std::micro>;
using milliSec = std::chrono::duration<float, std::milli>;
using Sec      = std::chrono::duration<float>;

template <typename DurationT>
GCXX_FH auto ConvertDuration(float ms) -> DurationT {
  return std::chrono::duration_cast<DurationT>(milliSec(ms));
}

class StreamView;

// Non-owning view of a GPU event; user creates and destroys it.
class EventView {

 public:
  using deviceEvent_t   = driver::deviceEvent_t;
  using raw_handle_type = driver::deviceEvent_t;

  EventView() = default;


  GCXX_CXPR EventView(deviceEvent_t rawEvent) GCXX_NOEXCEPT;

  GCXX_CXPR EventView(const EventView& eventRef) GCXX_NOEXCEPT;

  GCXX_CXPR
  auto operator=(const EventView& eventRef) GCXX_NOEXCEPT->EventView&;

  GCXX_FHC auto getRawHandle() GCXX_CONST_NOEXCEPT -> raw_handle_type;

  GCXX_CXPR explicit operator bool() GCXX_CONST_NOEXCEPT;

  GCXX_CXPR
  friend auto operator==(const EventView lhs,
                         const EventView rhs) GCXX_NOEXCEPT->bool;

  GCXX_CXPR
  friend auto operator!=(const EventView& lhs,
                         const EventView& rhs) GCXX_NOEXCEPT->bool;

  EventView(int) = delete;

  EventView(std::nullptr_t) = delete;

  GCXX_FH auto hasOccurred() const -> bool;

  GCXX_FH auto sync() const -> void;

  GCXX_FH auto recordInStream(
    flags::eventRecord recordFlag = flags::eventRecord::None) -> void;

  GCXX_FH auto recordInStream(
    const StreamView& stream,
    flags::eventRecord recordFlag = flags::eventRecord::None) -> void;

  // Both events must have been recorded before this call.
  template <typename DurationT = Sec>
  GCXX_FH auto elapsedTimeSince(const EventView& startEvent) const -> DurationT;

  // Both events must have been recorded before this call.
  template <typename DurationT = Sec>
  GCXX_FH static auto elapsedTimeBetween(
    const EventView& startEvent, const EventView& endEvent) -> DurationT;

 protected:
  deviceEvent_t m_event{driver::INVALID_EVENT};  // NOLINT
};

GCXX_NAMESPACE_MAIN_END()

#include <gcxx/runtime/details/event/event_view.inl>


#endif
