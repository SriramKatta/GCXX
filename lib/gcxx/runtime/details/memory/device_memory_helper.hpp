// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_MEMORY_DEVICE_MEMORY_HELPER_HPP
#define GCXX_RUNTIME_DETAILS_MEMORY_DEVICE_MEMORY_HELPER_HPP

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/error/runtime_error.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>
#include <gcxx/runtime_backend/backend_memory.hpp>

GCXX_NAMESPACE_MAIN_DETAILS_BEGIN()

struct device_malloc_t {
  auto operator()(std::size_t numbytes) const {
    return driver::deviceMalloc(numbytes);
  }
};

struct device_malloc_async_t {
  auto operator()(std::size_t numbytes,
                  const StreamView& sv = StreamView::Null()) const {
    return driver::deviceMallocAsync(numbytes, sv.getRawStream());
  }
};

struct device_managed_malloc_t {
  auto operator()(std::size_t numbytes) const {
    return driver::deviceMallocManaged(numbytes);
  }
};

struct host_malloc_t {
  auto operator()(std::size_t numbytes) const {
    return driver::deviceMallocHost(numbytes);
  }
};

struct device_free_t {
  auto operator()(void* ptr) const { driver::deviceFree(ptr); }
};

struct device_free_async_t {
  auto operator()(void* ptr, const StreamView& sv = StreamView::Null()) const {
    driver::deviceFreeAsync(ptr, sv.getRawStream());
  }
};

struct host_free_t {
  auto operator()(void* ptr) const { driver::deviceFreeHost(ptr); }
};

GCXX_NAMESPACE_MAIN_DETAILS_END()

GCXX_NAMESPACE_MAIN_BEGIN()

inline auto device_malloc         = details_::device_malloc_t{};
inline auto device_managed_malloc = details_::device_managed_malloc_t{};
inline auto device_malloc_async   = details_::device_malloc_async_t{};
inline auto device_free           = details_::device_free_t{};
inline auto device_free_async     = details_::device_free_async_t{};
inline auto host_malloc           = details_::host_malloc_t{};
inline auto host_free             = details_::host_free_t{};

GCXX_NAMESPACE_MAIN_END()


#endif