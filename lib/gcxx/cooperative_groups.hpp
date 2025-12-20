#pragma once
#ifndef GCXX_COOPRATIVE_GROUPS_HPP_
#define GCXX_COOPRATIVE_GROUPS_HPP_

#if GCXX_CUDA_MODE
#include <cooperative_groups.h>
#elif GCXX_HIP_MODE
#include <hip/hip_cooperative_groups.h>
#endif

#endif