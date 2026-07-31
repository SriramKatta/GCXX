// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_BUFFERS_NO_INIT_HPP_
#define GCXX_RUNTIME_MEMORY_BUFFERS_NO_INIT_HPP_

#include <gcxx/internal/prologue.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()


/// Tag type selecting lazy (no-allocation) buffer construction.
struct no_init_t {
  explicit no_init_t() = default;
};

/// Tag instance for lazy buffer construction.
inline constexpr no_init_t no_init{};


GCXX_NAMESPACE_MAIN_END()

#endif
