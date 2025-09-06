#pragma once
#ifndef GPUCXX_RUNTIME_DETAILS_EVENT_REF_INL_
#define GPUCXX_RUNTIME_DETAILS_EVENT_REF_INL_

#include <gpucxx/backend/backend.hpp>
#include <gpucxx/macros/define_macros.hpp>
#include <gpucxx/runtime/__flags/eventflags.hpp>
#include <gpucxx/runtime/runtime_error.hpp>

#include <utility>

GPUCXX_BEGIN_NAMESPACE

GPUCXX_FH auto event_ref::HasOccurred() const -> bool {
  auto err            = GPUCXX_RUNTIME_BACKEND(EventQuery)(event_);
  constexpr auto line = __LINE__ - 1;
  switch (err) {
    case GPUCXX_RUNTIME_BACKEND(Success):
      return true;
    case GPUCXX_RUNTIME_BACKEND(ErrorNotReady):
      return false;
    default:
      details_::checkDeviceError(err, "event Query", __FILE__, line);
      return false;
  }
}

GPUCXX_FH auto event_ref::RecordInStream(
  const stream_ref& stream, const flags::eventRecord recordFlag) -> void {
  GPUCXX_SAFE_RUNTIME_CALL(
    EventRecordWithFlags,
    (event_, stream.get(), static_cast<flag_t>(recordFlag)));
}

GPUCXX_FH auto event_ref::Synchronize() const -> void {
  GPUCXX_SAFE_RUNTIME_CALL(EventSynchronize, (event_));
}

template <typename DurationT>
GPUCXX_FH auto event_ref::ElapsedTimeSince(const event_ref& startEvent) const
  -> DurationT {
  this->Synchronize();
  float ms{};
  GPUCXX_SAFE_RUNTIME_CALL(EventElapsedTime,
                           (&ms, startEvent.get(), this->get()));
  return details_::ConvertDuration<DurationT>(ms);
}

GPUCXX_END_NAMESPACE


#endif
