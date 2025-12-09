#pragma once
#ifndef GCXX_RUNTIME_DETAILS_EVENT_REF_INL_
#define GCXX_RUNTIME_DETAILS_EVENT_REF_INL_

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>
#include <gcxx/runtime/flags/eventflags.hpp>
#include <gcxx/runtime/runtime_error.hpp>

#include <utility>

GCXX_NAMESPACE_MAIN_BEGIN

GCXX_FH auto event_wrap::HasOccurred() const -> bool {
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

GCXX_FH auto event_wrap::RecordInStream(const stream_wrap& stream,
                                        const flags::eventRecord recordFlag)
  -> void {
  GCXX_SAFE_RUNTIME_CALL(EventRecordWithFlags,
                         "Failed to recoed GPU Event in GPU Stream", event_,
                         stream.get(), static_cast<flag_t>(recordFlag));
}

GCXX_FH auto event_wrap::Synchronize() const -> void {
  GCXX_SAFE_RUNTIME_CALL(EventSynchronize, "Failed to synchronize GPU Event",
                         event_);
}

template <typename DurationT>
GCXX_FH auto event_wrap::ElapsedTimeSince(const event_wrap& startEvent) const
  -> DurationT {
  this->Synchronize();
  float ms{};
  GCXX_SAFE_RUNTIME_CALL(EventElapsedTime,
                         "Failed to get elapsed time between GPU Events", &ms,
                         startEvent.get(), this->get());
  return ConvertDuration<DurationT>(ms);
}

GCXX_NAMESPACE_MAIN_END


#endif
