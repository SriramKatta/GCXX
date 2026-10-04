// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_SCALARS_DEFAULT_SCALAR_POOLS_HPP_
#define GCXX_RUNTIME_MEMORY_SCALARS_DEFAULT_SCALAR_POOLS_HPP_

#include <utility>

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/memory/mempool/default_memory_pools.hpp>
#include <gcxx/runtime/memory/scalars/scalar.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

// Default-pool convenience factories for the owning scalars. The
// explicit-resource form is make_scalar<VT, Properties...> / the scalar ctors
// (see scalar.hpp); these mirror how callers already pair buffers with the
// default pools (examples/vector_add).

template <typename VT, typename... Args>
GCXX_FH auto make_device_scalar(const gcxx::DeviceHandle& device,
                                gcxx::StreamView stream,
                                Args&&... args) -> device_scalar<VT> {
  return device_scalar<VT>{stream, device_default_memory_pool(device),
                           std::forward<Args>(args)...};
}

template <typename VT, typename... Args>
GCXX_FH auto make_pinned_scalar(gcxx::StreamView stream,
                                Args&&... args) -> pinned_scalar<VT> {
  return pinned_scalar<VT>{stream, pinned_default_memory_pool(),
                           std::forward<Args>(args)...};
}

#if GCXX_HAS_MANAGED_POOLS
template <typename VT, typename... Args>
GCXX_FH auto make_managed_scalar(gcxx::StreamView stream,
                                 Args&&... args) -> managed_scalar<VT> {
  return managed_scalar<VT>{stream, managed_default_memory_pool(),
                            std::forward<Args>(args)...};
}
#endif  // GCXX_HAS_MANAGED_POOLS

GCXX_NAMESPACE_MAIN_END()

#endif
