#pragma once
#ifndef GPUCXX_RUNTIME_DETAILS_STREAM_INL_
#define GPUCXX_RUNTIME_DETAILS_STREAM_INL_

#include <gpucxx/runtime/__device/ensure_device.hpp>
#include <gpucxx/runtime/stream.hpp>

GPUCXX_BEGIN_NAMESPACE

GPUCXX_FH Stream::Stream(const flags::streamType createFlag,
                         const flags::streamPriority priorityFlag)
    : stream_ref(details_::__null_stream_) {
  if (createFlag == flags::streamType::nullStream) {
    return;
  }
  GPUCXX_SAFE_RUNTIME_CALL(StreamCreateWithPriority,
                           (&stream_, static_cast<flag_t>(createFlag),
                            -static_cast<flag_t>(priorityFlag)));
}

GPUCXX_FH auto Stream::destroy() -> void {
// since cudaStreamDestroy releases the handle after all work is done, to keep similar behaviour
#if GPUCXX_HIP_MODE
  Synchronize();
#endif

  if (stream_ == details_::__null_stream_ ||
      stream_ == details_::__invalid_stream_) {
    return;
  }
  int deviceId = -1;
  GPUCXX_SAFE_RUNTIME_CALL(StreamGetDevice, (stream_, &deviceId));
  details_::__EnsureCurrentDevice e(deviceId);
  GPUCXX_SAFE_RUNTIME_CALL(StreamDestroy, (stream_));
  stream_ = details_::__invalid_stream_;
}

GPUCXX_FH Stream::~Stream() {

  this->destroy();
}

GPUCXX_FH auto Stream::Create(const flags::streamType createFlag,
                              const flags::streamPriority priorityFlag)
  -> Stream {
  return {createFlag, priorityFlag};
}

GPUCXX_FH auto Stream::release() GPUCXX_NOEXCEPT -> stream_ref {
  auto oldStream = stream_;
  stream_        = details_::__invalid_stream_;
  return {oldStream};
}

GPUCXX_END_NAMESPACE


#endif