// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_SMARTPOINTERS_POINTERS_HPP_
#define GCXX_RUNTIME_SMARTPOINTERS_POINTERS_HPP_

#include <functional>
#include <memory>


#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/memory/device_memory_helper.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_NAMESPACE_MEMORY_BEGIN()

template <typename VT, typename DT>
using gcxx_unique_ptr = std::unique_ptr<VT, DT>;

template <typename VT>
using device_ptr = gcxx_unique_ptr<VT, std::function<void(VT*)>>;

template <typename VT>
using host_pinned_ptr = gcxx_unique_ptr<VT, decltype(host_free)>;  // NOLINT

template <typename VT>
using device_managed_ptr = gcxx_unique_ptr<VT, decltype(device_free)>;

template <typename VT>
auto make_device_unique_ptr(std::size_t numElem) -> device_ptr<VT> {
  return device_ptr<VT>{static_cast<VT*>(device_malloc(numElem * sizeof(VT))),
                        [](VT* p) {
                          device_free(static_cast<void*>(p));
                        }};
}

template <typename VT>
auto make_device_unique_ptr(const StreamView& sv,
                            std::size_t numElem) -> device_ptr<VT> {
  return device_ptr<VT>{
    static_cast<VT*>(device_malloc_async(numElem * sizeof(VT), sv)),
    [sv](VT* p) {
      device_free_async(static_cast<void*>(p), sv);
    }};
}

template <typename VT>
auto make_host_pinned_unique_ptr(std::size_t numElem) -> host_pinned_ptr<VT> {
  return host_pinned_ptr<VT>{
    static_cast<VT*>(host_malloc(numElem * sizeof(VT))), host_free};
}

template <typename VT>
auto make_device_managed_unique_ptr(std::size_t numElem)
  -> device_managed_ptr<VT> {
  return device_managed_ptr<VT>{
    static_cast<VT*>(device_managed_malloc(numElem * sizeof(VT))), device_free};
}
GCXX_NAMESPACE_MEMORY_END()

GCXX_NAMESPACE_MAIN_END()


#endif