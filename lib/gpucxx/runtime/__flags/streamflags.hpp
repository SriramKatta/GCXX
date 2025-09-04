#pragma once
#ifndef GPUCXX_RUNTIME_FLAGS_STREAMFLAGS_HPP_
#define GPUCXX_RUNTIME_FLAGS_STREAMFLAGS_HPP_

#include <gpucxx/backend/backend.hpp>
#include <gpucxx/macros/define_macros.hpp>

#include <limits>

GPUCXX_BEGIN_NAMESPACE

namespace flags {
  enum class streamType : flag_t {
    none        = GPUCXX_RUNTIME_BACKEND(StreamDefault),
    nonBlocking = GPUCXX_RUNTIME_BACKEND(StreamNonBlocking),
    // Sentinel value, chosen high to avoid clashes with future valid flags.
    null        = std::numeric_limits<flag_t>::max()
  };

  enum class streamPriority : flag_t {
    none      = 0U,
    veryLow  = 1U,
    low       = 2U,
    high      = 3U,
    veryHigh = 4U,
    critical  = 5U,
  };
}  // namespace flags

GPUCXX_END_NAMESPACE

#endif