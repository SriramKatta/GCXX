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

using f32_t = float;
using f64_t = double;

using cf32_t = std::complex<f32_t>;
using cf64_t = std::complex<f64_t>;

using i8_t  = std::int8_t;
using u8_t  = std::uint8_t;
using i32_t = std::int32_t;

GCXX_NAMESPACE_MAIN_END()

#endif
