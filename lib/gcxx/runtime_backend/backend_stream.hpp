// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_BACKEND_BACKEND_STREAM_HPP_
#define GCXX_RUNTIME_BACKEND_BACKEND_STREAM_HPP_

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime_backend/backend_handles.hpp>

GCXX_NAMESPACE_MAIN_DRIVER_BEGIN

// TODO : needed for device side stream allocation
// GCXX_FD auto streamCreate(unsigned int flags) -> deviceStream_t {
//   deviceStream_t stream{NULL_STREAM};
//   GCXX_SAFE_RUNTIME_CALL(StreamCreate, "Failed to Create Stream", &stream,
//                          flags);
//   return stream;
// }

GCXX_FH auto streamCreate(unsigned int flags, int priority) -> deviceStream_t {
  deviceStream_t stream{NULL_STREAM};
  GCXX_SAFE_RUNTIME_CALL(StreamCreateWithPriority, "Failed to Create Stream",
                         &stream, flags, -priority);
  return stream;
}

GCXX_FH auto streamDestroy(deviceStream_t stream) -> void {
  GCXX_SAFE_RUNTIME_CALL(StreamDestroy, "Failed to Destroy Stream", stream);
}

GCXX_FH auto streamSynchronize(deviceStream_t stream) -> void {
  GCXX_SAFE_RUNTIME_CALL(StreamSynchronize, "Failed to synchronize Stream",
                         stream);
}

GCXX_FH auto streamQueryNothrow(deviceStream_t stream)
  -> driver::deviceError_t {
  return GCXX_RUNTIME_BACKEND(StreamQuery)(stream);
}

GCXX_FH auto StreamWaitEvent(deviceStream_t stream, deviceEvent_t event,
                             unsigned int waitFlag) -> void {
  GCXX_SAFE_RUNTIME_CALL(StreamWaitEvent,
                         "Failed to GPU Stream Wait on GPU Event", stream,
                         event, waitFlag);
}

GCXX_NAMESPACE_MAIN_DRIVER_END

#endif