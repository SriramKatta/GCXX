#pragma once
#ifndef GCXX_RUNTIME_DETAILS_STREAM_VIEW_INL_
#define GCXX_RUNTIME_DETAILS_STREAM_VIEW_INL_

#include <gcxx/macros/define_macros.hpp>
#include <gcxx/runtime/runtime_error.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

  GCXX_FHC StreamView::StreamView(deviceStream_t rawStream) GCXX_NOEXCEPT
      : stream_(rawStream) {}

  GCXX_FH constexpr auto StreamView::getRawStream() GCXX_CONST_NOEXCEPT->deviceStream_t {
    return stream_;
  }

  GCXX_FH constexpr StreamView::operator deviceStream_t() GCXX_CONST_NOEXCEPT {
    return getRawStream();
  }

GCXX_FH auto StreamView::HasPendingWork() -> bool {
  auto err = GCXX_RUNTIME_BACKEND(StreamQuery)(stream_);
  switch (err) {
    case GCXX_RUNTIME_BACKEND(Success):
      return false;
    case GCXX_RUNTIME_BACKEND(ErrorNotReady):
      return true;
    default:
      details_::throwGPUError(err, "Failed to query GPU Stream");
  }
  return false;
}

GCXX_FH auto StreamView::Synchronize() const -> void {
  GCXX_SAFE_RUNTIME_CALL(StreamSynchronize, "Failed to synchronize GPU Stream",
                         stream_);
}

GCXX_FH auto StreamView::WaitOnEvent(const EventView& event,
                                     const flags::eventWait waitFlag) const
  -> void {
  GCXX_SAFE_RUNTIME_CALL(
    StreamWaitEvent, "Failed to GPU Stream Wait on GPU Event",
    this->getRawStream(), event.getRawEvent(), static_cast<flag_t>(waitFlag));
}

GCXX_NAMESPACE_MAIN_END

#include <gcxx/macros/undefine_macros.hpp>

#endif