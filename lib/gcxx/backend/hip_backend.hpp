#pragma once
#ifndef GCXX_BACKEND_HIP_BACKEND_HPP_
#define GCXX_BACKEND_HIP_BACKEND_HPP_

#include <hip/hip_runtime.h>

#define RUNTIME_BACKEND hip
#define ATTRIBUTE_BACKEND hipDeviceAttribute
#define ATTRIBUTE_BACKEND_TYPE hipDeviceAttribute_t

#endif