
#pragma once
#ifndef GCXX_RUNTIME_FLAGS_OCCUPANCY_FLAGS_HPP_
#define GCXX_RUNTIME_FLAGS_OCCUPANCY_FLAGS_HPP_

#include <gcxx/internal/prologue.hpp>

GCXX_NAMESPACE_MAIN_FLAGS_BEGIN

enum class occupancyType : details_::flag_t {
  Default = GCXX_RUNTIME_BACKEND(OccupancyDefault),
  DisableCachingOverride =
    GCXX_RUNTIME_BACKEND(OccupancyDisableCachingOverride),
};


GCXX_NAMESPACE_MAIN_FLAGS_END

#endif