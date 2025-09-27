#pragma once
#ifndef GCXX_RUNTIME_DETAILS_EVENT_REF_INL_
#define GCXX_RUNTIME_DETAILS_EVENT_REF_INL_

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>
#include <gcxx/runtime/flags/eventflags.hpp>
#include <gcxx/runtime/runtime_error.hpp>

#include <utility>

GCXX_BEGIN_NAMESPACE

GCXX_FH auto event_ref::HasOccurred() const -> bool {
  auto err            = GCXX_RUNTIME_BACKEND(EventQuery)(event_);
  constexpr auto line = __LINE__ - 1;
  switch (err) {
    case GCXX_RUNTIME_BACKEND(Success):
      return true;
    case GCXX_RUNTIME_BACKEND(ErrorNotReady):
      return false;
    default:
      details_::checkDeviceError(err, "event Query", __FILE__, line);
      return false;
  }
}

GCXX_FH auto event_ref::RecordInStream(
  const stream_ref& stream, const flags::eventRecord recordFlag) -> void {
  GCXX_SAFE_RUNTIME_CALL(
    EventRecordWithFlags,
    (event_, stream.get(), static_cast<flag_t>(recordFlag)));
}

GCXX_FH auto event_ref::Synchronize() const -> void {
  GCXX_SAFE_RUNTIME_CALL(EventSynchronize, (event_));
}

template <typename DurationT>
GCXX_FH auto event_ref::ElapsedTimeSince(const event_ref& startEvent) const
  -> DurationT {
  this->Synchronize();
  float ms{};
  GCXX_SAFE_RUNTIME_CALL(EventElapsedTime,
                         (&ms, startEvent.get(), this->get()));
  return details_::ConvertDuration<DurationT>(ms);
}

GCXX_END_NAMESPACE


#endif
