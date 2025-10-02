#pragma once
#ifndef GCXX_RUNTIME_DETAILS_STREAM_REF_INL_
#define GCXX_RUNTIME_DETAILS_STREAM_REF_INL_

#include <gcxx/macros/define_macros.hpp>
#include <gcxx/runtime/runtime_error.hpp>
#include <gcxx/runtime/stream/stream_ref.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

GCXX_FH auto stream_ref::HasPendingWork() -> bool {
  auto err            = GCXX_RUNTIME_BACKEND(StreamQuery)(stream_);
  constexpr auto line = __LINE__ - 1;
  switch (err) {
    case GCXX_RUNTIME_BACKEND(Success):
      return false;
    case GCXX_RUNTIME_BACKEND(ErrorNotReady):
      return true;
    default:
      details_::checkDeviceError(err, "stream Query", __FILE__, line);
      return true;
  }
}

GCXX_FH auto stream_ref::Synchronize() const -> void {
  GCXX_SAFE_RUNTIME_CALL(StreamSynchronize, (stream_));
}

GCXX_FH auto stream_ref::WaitOnEvent(const details_::event_wrap& event,
                                     const flags::eventWait waitFlag) const
  -> void {
  GCXX_SAFE_RUNTIME_CALL(
    StreamWaitEvent, (this->get(), event.get(), static_cast<flag_t>(waitFlag)));
}

GCXX_NAMESPACE_MAIN_END

#include <gcxx/macros/undefine_macros.hpp>

#endif