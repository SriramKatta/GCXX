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
  };

  enum class eventRecord : flag_t {
    Default  = GPUCXX_RUNTIME_BACKEND(EventRecordDefault),
    External = GPUCXX_RUNTIME_BACKEND(EventRecordExternal),
  };
}  // namespace flags

GPUCXX_END_NAMESPACE

#endif