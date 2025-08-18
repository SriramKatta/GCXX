#pragma once
#ifndef GPUCXX_API_RUNTIME_FLAGS_STREAMFLAGS_HPP_
#define GPUCXX_API_RUNTIME_FLAGS_STREAMFLAGS_HPP_

#include <gpucxx/backend/backend.hpp>
#include <gpucxx/macros/define_macros.hpp>

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
    Default  = 0U,
    VeryLow  = 1U,
    Low      = 2U,
    High     = 3U,
    VeryHigh = 4U,
    Critical = 5U,
  };
}  // namespace flags

GPUCXX_END_NAMESPACE

#endif