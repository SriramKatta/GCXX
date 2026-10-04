// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_TYPES_SCALAR_TYPES_HPP
#define GCXX_TYPES_SCALAR_TYPES_HPP

#include <complex>
#include <cstdint>
#include <gcxx/internal/prologue.hpp>

// TODO: Deferred f16_t/bf16_t extension point; needs a backend-gated header.
GCXX_NAMESPACE_MAIN_BEGIN()

using float32_t = float;
using float64_t = double;

using cf32_t = std::complex<float32_t>;
using cf64_t = std::complex<float64_t>;

using int8_t  = std::int8_t;
using uint8_t = std::uint8_t;
using int32_t = std::int32_t;
using int64_t = std::int64_t;

GCXX_NAMESPACE_MAIN_END()

#endif
