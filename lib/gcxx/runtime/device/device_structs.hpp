#pragma once
#ifndef GCXX_RUNTIME_DEVICE_DEVICE_STRUCTS_HPP_
#define GCXX_RUNTIME_DEVICE_DEVICE_STRUCTS_HPP_

#include <string_view>

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>

#include <gcxx/runtime/memory/spans/spans.hpp>


GCXX_NAMESPACE_MAIN_BEGIN


#if GCXX_CUDA_MODE
using DeviceProp = cudaDeviceProp;
#elif GCXX_HIP_MODE
using DeviceProp = hipDeviceProp_t;
#else
#error "Some horrible UB is happening now"
#endif

GCXX_NAMESPACE_MAIN_END


#endif