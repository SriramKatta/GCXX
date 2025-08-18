#pragma once
#ifndef GPUCXX_RUNTIME_FLAGS_EVENTFLAGS_HPP_
#define GPUCXX_RUNTIME_FLAGS_EVENTFLAGS_HPP_

#include <gpucxx/backend/backend.hpp>
#include <gpucxx/macros/define_macros.hpp>

GPUCXX_BEGIN_NAMESPACE

namespace flags {
  enum class eventCreate : flag_t {
    none          = GPUCXX_RUNTIME_BACKEND(EventDefault),
    blockingSync  = GPUCXX_RUNTIME_BACKEND(EventBlockingSync),
    disableTiming = GPUCXX_RUNTIME_BACKEND(EventDisableTiming),
    interprocess  = GPUCXX_RUNTIME_BACKEND(EventInterprocess) |
                   GPUCXX_RUNTIME_BACKEND(EventDisableTiming),
  };

  enum class streamCapture : flag_t {
    add = GPUCXX_RUNTIME_BACKEND(StreamAddCaptureDependencies),
    set = GPUCXX_RUNTIME_BACKEND(StreamSetCaptureDependencies),
  };

  enum class eventRecord : flag_t {
    none     = GPUCXX_RUNTIME_BACKEND(EventRecordDefault),
    external = GPUCXX_RUNTIME_BACKEND(EventRecordExternal),
  };

  enum class eventWait : flag_t {
    none     = GPUCXX_RUNTIME_BACKEND(EventWaitDefault),
    external = GPUCXX_RUNTIME_BACKEND(EventWaitExternal),
  };
}  // namespace flags

GPUCXX_END_NAMESPACE

#endif