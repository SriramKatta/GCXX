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

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/memory/buffers/properties.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()


// ─────────────────────────────────────────────────────────────────────────────
// gcxx::memory resource concepts.
//
// A gcxx resource is duck-typed around the stream-ordered allocation contract:
//
//   void* allocate(gcxx::StreamView, std::size_t num_bytes);
//   void  deallocate(gcxx::StreamView, void* ptr);
//
// and advertises its accessibility via `using properties = TypeSet<...>;`.
// ─────────────────────────────────────────────────────────────────────────────

// resource_api<Resource>: true iff Resource exposes the stream-ordered
// allocate/deallocate members with the expected signatures. This is the
// duck-typed signature check, isolated so buffer can report a missing API
// distinctly from a missing property.
//
// allocate must yield void* (the gcxx resource contract).
// deallocate is only required to be callable.
template <typename Resource>
GCXX_CONCEPT resource_api = GCXX_REQUIRES_EXPR(
  (Resource), Resource& r, gcxx::StreamView stream, std::size_t num_bytes,
  void* ptr)(_Same_as(void*)(r.allocate(stream, num_bytes)),
             (r.deallocate(stream, ptr)));

// resource_with<Resource, Properties...>: a type with the resource API that
// advertises every one of Properties... via its `using properties` TypeSet.
// Mirrors CCCL's cuda::mr::resource_with (API + properties
template <typename Resource, typename... Properties>
GCXX_CONCEPT resource_with =
  resource_has_all_v<Resource, Properties...> && resource_api<Resource>;


GCXX_NAMESPACE_MAIN_END()

#endif
