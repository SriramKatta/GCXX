// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_MEMORY_MEMPOOL_MEMPOOL_VIEW_INL_
#define GCXX_RUNTIME_DETAILS_MEMORY_MEMPOOL_MEMPOOL_VIEW_INL_

#include <gcxx/internal/prologue.hpp>

#include <vector>

#include <gcxx/runtime/device/device_handle.hpp>
#include <gcxx/runtime/flags/memory_flags.hpp>
#include <gcxx/runtime/memory/mempool/mempool_props.hpp>
#include <gcxx/runtime_backend/backend_stream_memory.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_FH auto MemPoolView::getRawMemPool() const -> deviceMemPool_t {
  return m_pool;
};

GCXX_FH auto MemPoolView::GetDefaultMempool(const DeviceHandle& hand)
  -> MemPoolView {
  return hand.GetDefaultMemPool();
}

GCXX_FH auto MemPoolView::MallocFromPoolAsync(
  const StreamView& stream, std::size_t numBytes) const -> void* {
  return driver::deviceMallocFromPoolAsync(numBytes, m_pool,
                                           stream.getRawStream());
}

GCXX_FH auto MemPoolView::TrimTo(std::size_t minBytesToKeep) const -> void {
  driver::deviceMemPoolTrimTo(m_pool, minBytesToKeep);
}

GCXX_FH auto MemPoolView::SetAccess(const MemAccessDesc* descList,
                                    std::size_t count) -> void {
  if (count == 0) {
    return;
  }
  std::vector<driver::deviceMemAccessDesc_t> rawList(count);
  for (std::size_t i = 0; i < count; ++i) {
    rawList[i] = descList[i].getRawMemAccessDesc();
  }
  driver::deviceMemPoolSetAccess(m_pool, rawList.data(), count);
}

GCXX_FH auto MemPoolView::SetAccess(const MemAccessDesc& desc) -> void {
  auto raw = desc.getRawMemAccessDesc();
  driver::deviceMemPoolSetAccess(m_pool, &raw, 1);
}

GCXX_FH auto MemPoolView::GetAccess(const MemAccessDesc& location) const
  -> flags::MemAccessFlags {
  auto rawLoc   = location.getRawMemLocation();
  auto rawFlags = driver::deviceMemPoolGetAccess(m_pool, &rawLoc);
  return static_cast<flags::MemAccessFlags>(rawFlags);
}

GCXX_FH auto MemPoolView::ExportPointer(void* ptr)
  -> deviceMemPoolPtrExportData_t {
  return driver::deviceMemPoolExportPointer(ptr);
}

GCXX_FH auto MemPoolView::ImportPointer(
  deviceMemPoolPtrExportData_t* exportData) const -> void* {
  return driver::deviceMemPoolImportPointer(m_pool, exportData);
}

GCXX_FH auto MemPoolView::ExportToShareableHandle(
  void* shareableHandle, flags::MemAllocationHandle handleType,
  unsigned int handleFlags) const -> void {
  driver::deviceMemPoolExportToShareableHandle(
    shareableHandle, m_pool,
    static_cast<driver::deviceMemAllocationHandleType_t>(handleType),
    handleFlags);
}

GCXX_FH auto MemPoolView::ImportFromShareableHandle(
  void* shareableHandle, flags::MemAllocationHandle handleType,
  unsigned int handleFlags) -> MemPoolView {
  auto pool = driver::deviceMemPoolImportFromShareableHandle(
    shareableHandle,
    static_cast<driver::deviceMemAllocationHandleType_t>(handleType),
    handleFlags);
  return MemPoolView(pool);
}

#if GCXX_CUDA_VERSION_GREATER_EQUAL(13, 0, 0)
GCXX_FH auto MemPoolView::GetDefaultMemPoolByLocation(
  const MemAccessDesc& location, flags::MemAllocation type) -> MemPoolView {
  auto rawLoc  = location.getRawMemLocation();
  auto rawType = static_cast<driver::deviceMemAllocationType_t>(type);
  return MemPoolView(driver::deviceMemPoolGetDefaultMemPool(&rawLoc, rawType));
}

GCXX_FH auto MemPoolView::GetMemPoolByLocation(
  const MemAccessDesc& location, flags::MemAllocation type) -> MemPoolView {
  auto rawLoc  = location.getRawMemLocation();
  auto rawType = static_cast<driver::deviceMemAllocationType_t>(type);
  return MemPoolView(driver::deviceMemPoolGetMemPool(&rawLoc, rawType));
}

GCXX_FH auto MemPoolView::SetMemPoolByLocation(const MemAccessDesc& location,
                                               flags::MemAllocation type,
                                               MemPoolView pool) -> void {
  auto rawLoc  = location.getRawMemLocation();
  auto rawType = static_cast<driver::deviceMemAllocationType_t>(type);
  driver::deviceMemPoolSetMemPool(&rawLoc, rawType, pool.getRawMemPool());
}
#endif

GCXX_FH auto MemPoolView::SetFollowEventDependencies(bool state) -> void {
  int threshold = static_cast<int>(state);
  driver::deviceMemPoolSetAttribute(
    m_pool,
    static_cast<driver::deviceMemPoolAttr_t>(
      flags::MemPoolAttr::FollowEventDependencies),
    static_cast<void*>(&threshold));
}

GCXX_FH auto MemPoolView::SetAllowOpportunistic(bool state) -> void {
  int threshold = static_cast<int>(state);
  driver::deviceMemPoolSetAttribute(m_pool,
                                    static_cast<driver::deviceMemPoolAttr_t>(
                                      flags::MemPoolAttr::AllowOpportunistic),
                                    static_cast<void*>(&threshold));
}

GCXX_FH auto MemPoolView::SetAllowInternalDependencies(bool state) -> void {
  int threshold = static_cast<int>(state);
  driver::deviceMemPoolSetAttribute(
    m_pool,
    static_cast<driver::deviceMemPoolAttr_t>(
      flags::MemPoolAttr::AllowInternalDependencies),
    static_cast<void*>(&threshold));
}

GCXX_FH auto MemPoolView::SetReleaseThreshold(std::uint64_t threshold) -> void {
  driver::deviceMemPoolSetAttribute(m_pool,
                                    static_cast<driver::deviceMemPoolAttr_t>(
                                      flags::MemPoolAttr::ReleaseThreshold),
                                    static_cast<void*>(&threshold));
}

GCXX_FH auto MemPoolView::SetReservedMemCurrent(std::uint64_t threshold)
  -> void {
  driver::deviceMemPoolSetAttribute(m_pool,
                                    static_cast<driver::deviceMemPoolAttr_t>(
                                      flags::MemPoolAttr::ReservedMemCurrent),
                                    static_cast<void*>(&threshold));
}

GCXX_FH auto MemPoolView::SetReservedMemHigh(std::uint64_t threshold) -> void {
  driver::deviceMemPoolSetAttribute(m_pool,
                                    static_cast<driver::deviceMemPoolAttr_t>(
                                      flags::MemPoolAttr::ReservedMemHigh),
                                    static_cast<void*>(&threshold));
}

GCXX_FH auto MemPoolView::SetUsedMemCurrent(std::uint64_t threshold) -> void {
  driver::deviceMemPoolSetAttribute(m_pool,
                                    static_cast<driver::deviceMemPoolAttr_t>(
                                      flags::MemPoolAttr::UsedMemCurrent),
                                    static_cast<void*>(&threshold));
}

GCXX_FH auto MemPoolView::SetUsedMemHigh(std::uint64_t threshold) -> void {
  driver::deviceMemPoolSetAttribute(
    m_pool,
    static_cast<driver::deviceMemPoolAttr_t>(flags::MemPoolAttr::UsedMemHigh),
    static_cast<void*>(&threshold));
}

GCXX_FH auto MemPoolView::GetFollowEventDependencies() -> bool {
  int retval{};
  driver::deviceMemPoolGetAttribute(
    m_pool,
    static_cast<driver::deviceMemPoolAttr_t>(
      flags::MemPoolAttr::FollowEventDependencies),
    static_cast<void*>(&retval));
  return static_cast<bool>(retval);
}

GCXX_FH auto MemPoolView::GetAllowOpportunistic() -> bool {
  int retval{};
  driver::deviceMemPoolGetAttribute(m_pool,
                                    static_cast<driver::deviceMemPoolAttr_t>(
                                      flags::MemPoolAttr::AllowOpportunistic),
                                    static_cast<void*>(&retval));
  return static_cast<bool>(retval);
}

GCXX_FH auto MemPoolView::GetAllowInternalDependencies() -> bool {
  int retval{};
  driver::deviceMemPoolGetAttribute(
    m_pool,
    static_cast<driver::deviceMemPoolAttr_t>(
      flags::MemPoolAttr::AllowInternalDependencies),
    static_cast<void*>(&retval));
  return static_cast<bool>(retval);
}

GCXX_FH auto MemPoolView::GetReleaseThreshold() -> std::uint64_t {
  std::uint64_t retval{};
  driver::deviceMemPoolGetAttribute(m_pool,
                                    static_cast<driver::deviceMemPoolAttr_t>(
                                      flags::MemPoolAttr::ReleaseThreshold),
                                    static_cast<void*>(&retval));
  return retval;
}

GCXX_FH auto MemPoolView::GetReservedMemCurrent() -> std::uint64_t {
  std::uint64_t retval{};
  driver::deviceMemPoolGetAttribute(m_pool,
                                    static_cast<driver::deviceMemPoolAttr_t>(
                                      flags::MemPoolAttr::ReservedMemCurrent),
                                    static_cast<void*>(&retval));
  return retval;
}

GCXX_FH auto MemPoolView::GetReservedMemHigh() -> std::uint64_t {
  std::uint64_t retval{};
  driver::deviceMemPoolGetAttribute(m_pool,
                                    static_cast<driver::deviceMemPoolAttr_t>(
                                      flags::MemPoolAttr::ReservedMemHigh),
                                    static_cast<void*>(&retval));
  return retval;
}

GCXX_FH auto MemPoolView::GetUsedMemCurrent() -> std::uint64_t {
  std::uint64_t retval{};
  driver::deviceMemPoolGetAttribute(m_pool,
                                    static_cast<driver::deviceMemPoolAttr_t>(
                                      flags::MemPoolAttr::UsedMemCurrent),
                                    static_cast<void*>(&retval));
  return retval;
}

GCXX_FH auto MemPoolView::GetUsedMemHigh() -> std::uint64_t {
  std::uint64_t retval{};
  driver::deviceMemPoolGetAttribute(
    m_pool,
    static_cast<driver::deviceMemPoolAttr_t>(flags::MemPoolAttr::UsedMemHigh),
    static_cast<void*>(&retval));
  return retval;
}

GCXX_NAMESPACE_MAIN_END()


#endif