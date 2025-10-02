#pragma once
#ifndef GCXX_RUNTIME_FLAGS_EVENTFLAGS_HPP_
#define GCXX_RUNTIME_FLAGS_EVENTFLAGS_HPP_

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

namespace flags {
  enum class eventCreate : flag_t {
    none          = GCXX_RUNTIME_BACKEND(EventDefault),
    blockingSync  = GCXX_RUNTIME_BACKEND(EventBlockingSync),
    disableTiming = GCXX_RUNTIME_BACKEND(EventDisableTiming),
    interprocess  = GCXX_RUNTIME_BACKEND(EventInterprocess) |
                   GCXX_RUNTIME_BACKEND(EventDisableTiming),
  };

  inline eventCreate operator|(const eventCreate& lhs, const eventCreate& rhs) {
    return static_cast<eventCreate>(static_cast<flag_t>(lhs) |
                                    static_cast<flag_t>(rhs));
  }

  enum class streamCapture : flag_t {
    add = GCXX_RUNTIME_BACKEND(StreamAddCaptureDependencies),
    set = GCXX_RUNTIME_BACKEND(StreamSetCaptureDependencies),
  };

  enum class eventRecord : flag_t {
    none     = GCXX_RUNTIME_BACKEND(EventRecordDefault),
    external = GCXX_RUNTIME_BACKEND(EventRecordExternal),
  };

  enum class eventWait : flag_t {
#if defined(GCXX_CUDA_MODE)
    none     = GCXX_RUNTIME_BACKEND(EventWaitDefault),
    external = GCXX_RUNTIME_BACKEND(EventWaitExternal),
#else  // its stupid!! these are supposedly defined as per documentation but not
       // implemented in the actual code
    none     = 0,
    external = 0,
#endif
  };
}  // namespace flags

GCXX_NAMESPACE_MAIN_END

#endif