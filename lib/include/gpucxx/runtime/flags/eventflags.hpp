#pragma once
#ifndef GPUCXX_RUNTIME_FLAGS_EVENT_HPP
#define GPUCXX_RUNTIME_FLAGS_EVENT_HPP

#include <gpucxx/utils/define_specifiers.hpp>

GPUCXX_BEGIN_NAMESPACE

namespace flags {
  enum class eventCreate : flag_t {
    Default       = GPUCXX_RUNTIME_BACKEND(EventDefault),
    BlockingSync  = GPUCXX_RUNTIME_BACKEND(EventBlockingSync),
    DisableTiming = GPUCXX_RUNTIME_BACKEND(EventDisableTiming),
    Interprocess  = GPUCXX_RUNTIME_BACKEND(EventInterprocess) |
                   GPUCXX_RUNTIME_BACKEND(EventDisableTiming),
#if GPUCXX_HIP_MODE
    DisableSystemFence = GPUCXX_RUNTIME_BACKEND(EventDisableSystemFence),
#endif
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
    Default  = GPUCXX_RUNTIME_BACKEND(EventWaitDefault),
    External = GPUCXX_RUNTIME_BACKEND(EventWaitExternal),
  };
}  // namespace flags

GPUCXX_END_NAMESPACE

#endif