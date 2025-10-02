#pragma once
#ifndef GCXX_RUNTIME_DETAILS_STREAM_INL_
#define GCXX_RUNTIME_DETAILS_STREAM_INL_

#include <gcxx/runtime/device/ensure_device.hpp>
#include <gcxx/runtime/stream.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

GCXX_FH Stream::Stream(const flags::streamType createFlag,
                       const flags::streamPriority priorityFlag)
    : stream_wrap(details_::NULL_STREAM) {
  if (createFlag == flags::streamType::nullStream) {
    return;
  }
  GCXX_SAFE_RUNTIME_CALL(StreamCreateWithPriority,
                         (&stream_, static_cast<flag_t>(createFlag),
                          -static_cast<flag_t>(priorityFlag)));
}

GCXX_FH auto Stream::destroy() -> void {
// since cudaStreamDestroy releases the handle after all work is done, to keep
// similar behaviour
#if GCXX_HIP_MODE
  Synchronize();
#endif

  if (stream_ == details_::NULL_STREAM || stream_ == details_::INVALID_STREAM) {
    return;
  }
  // int deviceId = -1;
  // GCXX_SAFE_RUNTIME_CALL(StreamGetDevice, (stream_, &deviceId));
  // details_::EnsureCurrentDevice e(deviceId);
  GCXX_SAFE_RUNTIME_CALL(StreamDestroy, (stream_));
  stream_ = details_::INVALID_STREAM;
}

GCXX_FH Stream::~Stream() {

  this->destroy();
}

GCXX_FH auto Stream::Create(const flags::streamType createFlag,
                            const flags::streamPriority priorityFlag)
  -> Stream {
  return {createFlag, priorityFlag};
}

GCXX_FH auto Stream::release() GCXX_NOEXCEPT -> stream_wrap {
  auto oldStream = stream_;
  stream_        = details_::INVALID_STREAM;
  return {oldStream};
}

GCXX_NAMESPACE_MAIN_END


#endif