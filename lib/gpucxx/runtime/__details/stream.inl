#pragma once
#ifndef GPUCXX_RUNTIME_DETAILS_STREAM_INL_
#define GPUCXX_RUNTIME_DETAILS_STREAM_INL_

#include <gpucxx/runtime/stream.hpp>

GPUCXX_BEGIN_NAMESPACE

GPUCXX_FH Stream::Stream(const flags::streamBehaviour createFlag,
                         const flags::streamPriority priorityFlag)
    : stream_ref(details_::__null_stream_) {
  if (createFlag == flags::streamBehaviour::none) {
    return;
  }
  GPUCXX_SAFE_RUNTIME_CALL(StreamCreateWithPriority,
                           (&stream_, static_cast<flag_t>(createFlag),
                            -static_cast<flag_t>(priorityFlag)));
}

GPUCXX_FH auto Stream::destroy() -> void {
  if (stream_ != details_::__null_stream_ ||
      stream_ != details_::__invalid_stream_) {
    GPUCXX_SAFE_RUNTIME_CALL(StreamDestroy, (stream_));
    stream_ = details_::__invalid_stream_;
  }
}

GPUCXX_FH Stream::~Stream() {
  this->destroy();
}

GPUCXX_FH auto Stream::Create(const flags::streamBehaviour createFlag,
                              const flags::streamPriority priorityFlag)
  -> Stream {
  return {createFlag, priorityFlag};
}

GPUCXX_FH auto Stream::release() GPUCXX_NOEXCEPT -> stream_ref {
  auto oldStream = stream_;
  stream_        = details_::__null_stream_;
  return stream_ref(oldStream);
}

GPUCXX_END_NAMESPACE


#endif