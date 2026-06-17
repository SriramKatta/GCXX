// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMEORY_MEMPOOL_MEMPOOL_VIEW_HPP_
#define GCXX_RUNTIME_MEMEORY_MEMPOOL_MEMPOOL_VIEW_HPP_

#include <gcxx/internal/prologue.hpp>

#include <cstdint>

#include <gcxx/runtime/device/device_handle.hpp>
#include <gcxx/runtime/flags/memory_flags.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

class MemPoolView {
 public:
  using deviceMemPool_t = driver::deviceMemPool_t;

  MemPoolView() = default;

  MemPoolView(deviceMemPool_t pool) : m_pool(pool) {}

  GCXX_FH auto getRawMemPool() const -> deviceMemPool_t;

  GCXX_FH static auto GetDefaultMempool(const DeviceHandle&) -> MemPoolView;

  GCXX_FH auto SetFollowEventDependencies(bool state) -> void;
  GCXX_FH auto SetAllowOpportunistic(bool state) -> void;
  GCXX_FH auto SetAllowInternalDependencies(bool state) -> void;
  GCXX_FH auto SetReleaseThreshold(std::uint64_t threshold) -> void;
  GCXX_FH auto SetReservedMemCurrent(std::uint64_t threshold) -> void;
  GCXX_FH auto SetReservedMemHigh(std::uint64_t threshold) -> void;
  GCXX_FH auto SetUsedMemCurrent(std::uint64_t threshold) -> void;
  GCXX_FH auto SetUsedMemHigh(std::uint64_t threshold) -> void;

  GCXX_FH auto GetFollowEventDependencies() -> bool;
  GCXX_FH auto GetAllowOpportunistic() -> bool;
  GCXX_FH auto GetAllowInternalDependencies() -> bool;
  GCXX_FH auto GetReleaseThreshold() -> std::uint64_t;
  GCXX_FH auto GetReservedMemCurrent() -> std::uint64_t;
  GCXX_FH auto GetReservedMemHigh() -> std::uint64_t;
  GCXX_FH auto GetUsedMemCurrent() -> std::uint64_t;
  GCXX_FH auto GetUsedMemHigh() -> std::uint64_t;

 protected:
  deviceMemPool_t m_pool{
    nullptr};  // NOLINT(cppcoreguidelines-non-private-member-variables-in-classes)
};

GCXX_NAMESPACE_MAIN_END()


#include <gcxx/runtime/details/memory/mempool/mempool_view.inl>

#endif