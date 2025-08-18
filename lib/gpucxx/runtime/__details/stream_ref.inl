#pragma once
#ifndef GPUCXX_API_DETAILS_STREAM_INL_
#define GPUCXX_API_DETAILS_STREAM_INL_

#include <gpucxx/macros/define_macros.hpp>
#include <gpucxx/runtime/__stream/stream_ref.hpp>

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

GPUCXX_END_NAMESPACE

#include <gpucxx/macros/undefine_macros.hpp>

#endif