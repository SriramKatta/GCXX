#pragma once
#ifndef GPUCXX_RUNTIME_FLAGS_EVENT_HPP
#define GPUCXX_RUNTIME_FLAGS_EVENT_HPP

#include <gpucxx/backend/backend.hpp>
#include <gpucxx/utils/define_specifiers.hpp>

GPUCXX_BEGIN_NAMESPACE

namespace flags {
  enum class eventCreate : flag_t {
    Default       = GPUCXX_RUNTIME_BACKEND(EventDefault),
    BlockingSync  = GPUCXX_RUNTIME_BACKEND(EventBlockingSync),
    DisableTiming = GPUCXX_RUNTIME_BACKEND(EventDisableTiming),
    Interprocess  = GPUCXX_RUNTIME_BACKEND(EventInterprocess) | GPUCXX_RUNTIME_BACKEND(EventDisableTiming),
  };

  enum class streamCapture : flag_t {
    Add = GPUCXX_RUNTIME_BACKEND(StreamAddCaptureDependencies),
    Set = GPUCXX_RUNTIME_BACKEND(StreamSetCaptureDependencies),
  };

  enum class eventRecord : flag_t {
    Default  = GPUCXX_RUNTIME_BACKEND(EventRecordDefault),
    External = GPUCXX_RUNTIME_BACKEND(EventRecordExternal),
  };

  enum class eventWait : flag_t {
#if GPUCXX_CUDA_MODE
    Default  = GPUCXX_RUNTIME_BACKEND(EventWaitDefault),
    External = GPUCXX_RUNTIME_BACKEND(EventWaitExternal),
#else
    Default  = 0,
    External = 0,
#endif
  };
}  // namespace flags

GPUCXX_END_NAMESPACE

#endif