// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_MEMPOOL_MEMPOOL_VIEW_HPP_
#define GCXX_RUNTIME_MEMORY_MEMPOOL_MEMPOOL_VIEW_HPP_

#include <gcxx/internal/prologue.hpp>

#include <cstddef>
#include <cstdint>

#include <gcxx/runtime/device/device_handle.hpp>
#include <gcxx/runtime/flags/memory_flags.hpp>
#include <gcxx/runtime/memory/mempool/mempool_props.hpp>
#include <gcxx/runtime/stream.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

class MemPoolView {
 public:
  using deviceMemPool_t = driver::deviceMemPool_t;

  MemPoolView() = default;

  MemPoolView(deviceMemPool_t pool) : m_pool(pool) {}

  GCXX_FH auto getRawMemPool() const -> deviceMemPool_t;

  GCXX_FH static auto GetDefaultMempool(const DeviceHandle&) -> MemPoolView;

  GCXX_FH auto MallocFromPoolAsync(const StreamView& stream,
                                   std::size_t numBytes) const -> void*;

  GCXX_FH auto TrimTo(std::size_t minBytesToKeep) const -> void;

  GCXX_FH auto SetAccess(const MemAccessDesc* descList,
                         std::size_t count) -> void;
  GCXX_FH auto SetAccess(const MemAccessDesc& desc) -> void;
  GCXX_FH auto GetAccess(const MemAccessDesc& location) const
    -> flags::MemAccessFlags;

  // ── IPC / inter-process sharing ─────────────────────────────────────────
  // ExportPointer/ImportPointer share a single allocation across pool
  // instances within the same process. ExportToShareableHandle /
  // ImportFromShareableHandle share an entire pool across processes via an
  // OS handle (POSIX fd on Linux, Win32 HANDLE on Windows). The shareable
  // handle pointer is typed as void* to match the driver ABI.
  using deviceMemPoolPtrExportData_t = driver::deviceMemPoolPtrExportData_t;

  GCXX_FH static auto ExportPointer(void* ptr) -> deviceMemPoolPtrExportData_t;

  GCXX_FH auto ImportPointer(deviceMemPoolPtrExportData_t* exportData) const
    -> void*;

  GCXX_FH auto ExportToShareableHandle(void* shareableHandle,
                                       flags::MemAllocationHandle handleType,
                                       unsigned int handleFlags) const -> void;

  GCXX_FH static auto ImportFromShareableHandle(
    void* shareableHandle, flags::MemAllocationHandle handleType,
    unsigned int handleFlags) -> MemPoolView;

#if GCXX_CUDA_VERSION_GREATER_EQUAL(13, 0, 0)
  // ── CUDA 13+ location-keyed pool API ────────────────────────────────────
  GCXX_FH static auto GetDefaultMemPoolByLocation(
    const MemAccessDesc& location, flags::MemAllocation type) -> MemPoolView;
  GCXX_FH static auto GetMemPoolByLocation(
    const MemAccessDesc& location, flags::MemAllocation type) -> MemPoolView;
  GCXX_FH static auto SetMemPoolByLocation(const MemAccessDesc& location,
                                           flags::MemAllocation type,
                                           MemPoolView pool) -> void;
#endif

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