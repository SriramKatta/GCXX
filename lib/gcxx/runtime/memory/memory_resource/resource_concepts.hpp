// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Derived from NVIDIA CCCL libcudacxx:
//   https://github.com/NVIDIA/cccl/blob/main/libcudacxx/include/cuda/__memory_resource
// Copyright (c) NVIDIA Corporation and affiliates.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_MEMORY_RESOURCE_RESOURCE_CONCEPTS_HPP_
#define GCXX_RUNTIME_MEMORY_MEMORY_RESOURCE_RESOURCE_CONCEPTS_HPP_

#include <type_traits>

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/memory/buffers/properties.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()


// Resources are duck-typed on stream-ordered allocate/deallocate.

// Isolated signature check so a missing API is reported distinctly.
template <typename Resource>
GCXX_CONCEPT resource_api = GCXX_REQUIRES_EXPR(
  (Resource), Resource& r, gcxx::StreamView stream, std::size_t num_bytes,
  void* ptr)(_Same_as(void*)(r.allocate(stream, num_bytes)),
             (r.deallocate(stream, ptr)));

// Mirrors CCCL's cuda::mr::resource_with (API + properties).
template <typename Resource, typename... Properties>
GCXX_CONCEPT resource_with =
  resource_has_all_v<Resource, Properties...> && resource_api<Resource>;


GCXX_NAMESPACE_MAIN_END()

GCXX_NAMESPACE_MAIN_DETAILS_BEGIN()

// Shared ctor gate for the property-carrying containers (buffer, scalar):
// uniform static_asserts for every resource-taking constructor.
template <typename Resource, typename... Properties>
constexpr auto validate_resource() -> void {
  static_assert(
    std::is_copy_constructible_v<std::decay_t<Resource>>,
    "resource must be copy constructible: the container owns a copy of it");
  static_assert(resource_has_all_v<std::decay_t<Resource>, Properties...>,
                "resource properties do not satisfy the container's "
                "Properties (e.g. a host_accessible resource cannot back a "
                "device_accessible container)");
  static_assert(resource_api<std::decay_t<Resource>>,
                "resource does not model the gcxx resource concept: it must "
                "expose allocate(gcxx::StreamView, std::size_t) -> void* and "
                "deallocate(gcxx::StreamView, void*) -> void");
}

GCXX_NAMESPACE_MAIN_DETAILS_END()

#endif
