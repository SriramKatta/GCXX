#pragma once
#ifndef GPUCXX_RUNTIME_DETAILS_STREAM_REF_INL_
#define GPUCXX_RUNTIME_DETAILS_STREAM_REF_INL_

#include <gpucxx/macros/define_macros.hpp>
#include <gpucxx/runtime/__stream/stream_ref.hpp>
#include <gpucxx/runtime/runtime_error.hpp>

GPUCXX_BEGIN_NAMESPACE

GPUCXX_FH auto stream_ref::HasPendingWork() -> bool {
  auto err            = GPUCXX_RUNTIME_BACKEND(StreamQuery)(stream_);
  constexpr auto line = __LINE__ - 1;
  switch (err) {
    case GPUCXX_RUNTIME_BACKEND(Success):
      return false;
    case GPUCXX_RUNTIME_BACKEND(ErrorNotReady):
      return true;
    default:
      details_::checkDeviceError(err, "stream Query", __FILE__, line);
      return true;
  }
}

GPUCXX_FH auto stream_ref::Synchronize() const -> void {
  GPUCXX_SAFE_RUNTIME_CALL(StreamSynchronize, (stream_));
}

GPUCXX_FH auto stream_ref::WaitOnEvent(const details_::event_base& event,
                                        const flags::eventWait waitFlag) const
  -> void {
  GPUCXX_SAFE_RUNTIME_CALL(
    StreamWaitEvent, (this->get(), event.get(), static_cast<flag_t>(waitFlag)));
}

GPUCXX_END_NAMESPACE

#include <gpucxx/macros/undefine_macros.hpp>

#endif