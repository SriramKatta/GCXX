#pragma once
#ifndef GCXX_RUNTIME_LAUNCH_LAUNCH_KERNELS_HPP
#define GCXX_RUNTIME_LAUNCH_LAUNCH_KERNELS_HPP

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>

GCXX_NAMESPACE_MAIN_DETAILS_BEGIN
using deviceLaunchConfig_t = GCXX_RUNTIME_BACKEND(LaunchConfig_t);
GCXX_NAMESPACE_MAIN_DETAILS_END


GCXX_NAMESPACE_MAIN_BEGIN

namspace launch {}

GCXX_NAMESPACE_MAIN_END

#endif