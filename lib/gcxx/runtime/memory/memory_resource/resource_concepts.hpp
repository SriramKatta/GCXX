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

GCXX_NAMESPACE_MEMORY_BEGIN()

// ─────────────────────────────────────────────────────────────────────────────
// gcxx::memory resource concepts (CCCL cuda::mr::resource_with parity).
//
// A gcxx resource is duck-typed around the stream-ordered allocation contract:
//
//   void* allocate(std::size_t num_bytes, gcxx::StreamView);
//   void  deallocate(void* ptr,        gcxx::StreamView);
//
// and advertises its accessibility via `using properties = TypeSet<...>;`.
//
// Until now this contract was only enforced implicitly: buffer::validate_resource
// checked copy-constructibility and the advertised properties, but never the
// allocate/deallocate signature. These concepts — built on the GCXX concept DSL
// (GCXX_CONCEPT / GCXX_REQUIRES_EXPR, both C++20 and C++17-emulated) — finally
// make the contract checkable. resource_with<...> mirrors CCCL's
// cuda::mr::resource_with<Resource, Properties...>.
// ─────────────────────────────────────────────────────────────────────────────

// resource_api<Resource>: true iff Resource exposes the stream-ordered
// allocate/deallocate members with the expected signatures. This is the
// duck-typed signature check, isolated so buffer can report a missing API
// distinctly from a missing property.
//
// allocate must yield void* (the gcxx resource contract). deallocate is only
// required to be callable (return type unconstrained).
template <typename Resource>
GCXX_CONCEPT resource_api = GCXX_REQUIRES_EXPR(
  (Resource), Resource& r, std::size_t num_bytes, gcxx::StreamView stream,
  void* ptr)(_Same_as(void*)(r.allocate(num_bytes, stream)),
             (r.deallocate(ptr, stream)));

// resource_with<Resource, Properties...>: a type with the resource API that
// advertises every one of Properties... via its `using properties` TypeSet.
// Mirrors CCCL's cuda::mr::resource_with (API + properties — copyability is NOT
// part of the concept; buffer adds that separately in validate_resource, since
// it owns a copy of the resource). This is why the non-copyable owning pools
// still satisfy resource_with while only their *_ref views can back a buffer.
template <typename Resource, typename... Properties>
GCXX_CONCEPT resource_with =
  resource_has_all_v<Resource, Properties...> && resource_api<Resource>;

GCXX_NAMESPACE_MEMORY_END()

GCXX_NAMESPACE_MAIN_END()

#endif
