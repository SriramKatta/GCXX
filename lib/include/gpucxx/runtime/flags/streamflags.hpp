#pragma once
#ifndef GPUCXX_RUNTIME_FLAGS_STREAM_HPP
#define GPUCXX_RUNTIME_FLAGS_STREAM_HPP

#include <gpucxx/utils/define_specifiers.hpp>

#include <limits>

GPUCXX_BEGIN_NAMESPACE

namespace flags {
  enum class streamBehaviour : flag_t {
    Default     = GPUCXX_RUNTIME_BACKEND(StreamDefault),
    NonBlocking = GPUCXX_RUNTIME_BACKEND(StreamNonBlocking),
    Null        = std::numeric_limits<flag_t>::max()
    // Sentinel value, chosen high to avoid clashes with future valid flags.
  };

  enum class streamPriority : flag_t {
    lowest  = 0U,
    lower   = 1U,
    low     = 2U,
    medium  = 3U,
    high    = 4U,
    highest = 5U,
  };
}  // namespace flags

GPUCXX_END_NAMESPACE

#endif