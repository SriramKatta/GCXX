#pragma once
#ifndef GPUCXX_API_DETAILS_STREAM_INL_
#define GPUCXX_API_DETAILS_STREAM_INL_

#include <gpucxx/runtime/stream.hpp>

GPUCXX_BEGIN_NAMESPACE

GPUCXX_FH Stream::Stream(const flags::streamBehaviour createFlag,
                         const flags::streamPriority priorityFlag) {
  if (createFlag == flags::streamBehaviour::Null) {
    return;
  }
  GPUCXX_SAFE_RUNTIME_CALL(StreamCreateWithPriority,
                           (&stream_, static_cast<flag_t>(createFlag),
                            -static_cast<flag_t>(priorityFlag)));
}

GPUCXX_FH auto Stream::destroy() -> void {
  if (stream_) {
    GPUCXX_SAFE_RUNTIME_CALL(StreamDestroy, (stream_));
    stream_ = nullptr;
  }
}

GPUCXX_FH Stream::~Stream() {
  this->destroy();
}

GPUCXX_FH auto Stream::query() const -> bool {
  auto err            = GPUCXX_RUNTIME_BACKEND(StreamQuery)(stream_);
  constexpr auto line = __LINE__ - 1;
  switch (err) {
    case GPUCXX_RUNTIME_BACKEND(Success):
      return true;
    case GPUCXX_RUNTIME_BACKEND(ErrorNotReady):
      return false;
    default:
      details_::checkDeviceError(err, "stream Query", __FILE__, line);
      return false;
  }
}

GPUCXX_FH auto Stream::Synchronize() const -> void {
  GPUCXX_SAFE_RUNTIME_CALL(StreamSynchronize, (stream_));
}

GPUCXX_FH auto Stream::WaitOnEvent(const deviceRawEvent_t &event,
                                   const flags::eventWait waitFlag) const
  -> void {
  GPUCXX_SAFE_RUNTIME_CALL(StreamWaitEvent,
                           (stream_, event, static_cast<flag_t>(waitFlag)));
}

GPUCXX_FH auto Stream::Create(const flags::streamBehaviour createFlag,
                            const flags::streamPriority priorityFlag)
  -> Stream {
  return {createFlag, priorityFlag};
}

GPUCXX_END_NAMESPACE


#endif