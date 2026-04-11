// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BACKEND_CUDA_BACKEND_HPP_
#define GCXX_BACKEND_CUDA_BACKEND_HPP_

#include <cuda_runtime.h>

#define RUNTIME_BACKEND cuda
#define ATTRIBUTE_BACKEND cudaDevAttr
#define ATTRIBUTE_BACKEND_TYPE cudaDeviceAttr
#define LIMIT_BACKEND_TYPE cudaLimit


#endif