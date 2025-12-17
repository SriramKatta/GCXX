#pragma once
#ifndef GCXX_RUNTIME_DETAILS_EVENT_VIEW_INL_
#define GCXX_RUNTIME_DETAILS_EVENT_VIEW_INL_

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>
#include <gcxx/runtime/flags/eventflags.hpp>
#include <gcxx/runtime/runtime_error.hpp>

#include <gcxx/runtime/event/event_view.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>

#include <utility>

GCXX_NAMESPACE_MAIN_BEGIN

GCXX_CXPR EventView::EventView(deviceEvent_t rawEvent) GCXX_NOEXCEPT
    : event_(rawEvent) {}

GCXX_CXPR EventView::EventView(const EventView& eventRef) GCXX_NOEXCEPT
    : event_(eventRef.getRawEvent()) {}

GCXX_FHC auto EventView::getRawEvent() GCXX_CONST_NOEXCEPT -> deviceEvent_t {
  return event_;
}

GCXX_CXPR EventView::operator deviceEvent_t() GCXX_CONST_NOEXCEPT {
  return getRawEvent();
}

GCXX_CXPR EventView::operator bool() GCXX_CONST_NOEXCEPT {
  return event_ != details_::INVALID_EVENT;
}

GCXX_CXPR auto operator==(const EventView lhs,
                                            const EventView rhs) GCXX_NOEXCEPT
  -> bool {
  return lhs.event_ == rhs.event_;
}

GCXX_CXPR auto operator!=(const EventView& lhs,
                                            const EventView& rhs) GCXX_NOEXCEPT
  -> bool {
  return !(lhs == rhs);
}

GCXX_FH auto EventView::HasOccurred() const -> bool {
  auto err = GCXX_RUNTIME_BACKEND(EventQuery)(event_);
  switch (err) {
    case details_::deviceErrSuccess:
      return true;
    case details_::deviceErrNotReady:
      return false;
    default:
      details_::throwGPUError(err, "Failed to query GPU Event");
  }
  return false;
}

GCXX_FH auto EventView::RecordInStream(const flags::eventRecord recordFlag)
  -> void {
  GCXX_SAFE_RUNTIME_CALL(
    EventRecordWithFlags, "Failed to recoed GPU Event in GPU Stream", event_,
    details_::NULL_STREAM, static_cast<flag_t>(recordFlag));
}

GCXX_FH auto EventView::RecordInStream(const StreamView& stream,
                                       const flags::eventRecord recordFlag)
  -> void {
  GCXX_SAFE_RUNTIME_CALL(
    EventRecordWithFlags, "Failed to recoed GPU Event in GPU Stream", event_,
    stream.getRawStream(), static_cast<flag_t>(recordFlag));
}

GCXX_FH auto EventView::Synchronize() const -> void {
  GCXX_SAFE_RUNTIME_CALL(EventSynchronize, "Failed to synchronize GPU Event",
                         event_);
}

template <typename DurationT>
GCXX_FH auto EventView::ElapsedTimeSince(const EventView& startEvent) const
  -> DurationT {
  this->Synchronize();
  float ms{};
  GCXX_SAFE_RUNTIME_CALL(EventElapsedTime,
                         "Failed to get elapsed time between GPU Events", &ms,
                         startEvent.getRawEvent(), this->getRawEvent());
  return ConvertDuration<DurationT>(ms);
}

template <typename DurationT>
GCXX_FH auto EventView::ElapsedTimeBetween(const EventView& startEvent,
                                           const EventView& endEvent)
  -> DurationT {
  return endEvent.ElapsedTimeSince<DurationT>(startEvent);
}

GCXX_NAMESPACE_MAIN_END


#endif
