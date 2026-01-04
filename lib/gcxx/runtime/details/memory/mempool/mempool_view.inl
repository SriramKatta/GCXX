#pragma once
#ifndef GCXX_RUNTIME_DETAILS_MEMEORY_MEMPOOL_MEMPOOL_VIEW_INL_
#define GCXX_RUNTIME_DETAILS_MEMEORY_MEMPOOL_MEMPOOL_VIEW_INL_

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/device/device_handle.hpp>
#include <gcxx/runtime/flags/memory_flags.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

GCXX_FH auto MemPoolView::getRawMemPool() const -> deviceMemPool_t {
  return pool_;
};

GCXX_FH auto MemPoolView::GetDefaultMempool(const DeviceHandle& hand)
  -> MemPoolView {
  return hand.GetDefaultMemPool();
}

GCXX_FH auto MemPoolView::SetFollowEventDependencies(bool state) -> void {
  int threshold = static_cast<int>(state);
  GCXX_SAFE_RUNTIME_CALL(MemPoolSetAttribute,
                         "Failed to set FollowEventDependencies of mempool",
                         pool_,
                         static_cast<GCXX_RUNTIME_BACKEND(MemPoolAttr)>(
                           flags::MemPoolAttr::FollowEventDependencies),
                         static_cast<void*>(&threshold));
}

GCXX_FH auto MemPoolView::SetAllowOpportunistic(bool state) -> void {
  int threshold = static_cast<int>(state);
  GCXX_SAFE_RUNTIME_CALL(MemPoolSetAttribute,
                         "Failed to set AllowOpportunistic of mempool", pool_,
                         static_cast<GCXX_RUNTIME_BACKEND(MemPoolAttr)>(
                           flags::MemPoolAttr::AllowOpportunistic),
                         static_cast<void*>(&threshold));
}

GCXX_FH auto MemPoolView::SetAllowInternalDependencies(bool state) -> void {
  int threshold = static_cast<int>(state);
  GCXX_SAFE_RUNTIME_CALL(MemPoolSetAttribute,
                         "Failed to set AllowInternalDependencies of mempool",
                         pool_,
                         static_cast<GCXX_RUNTIME_BACKEND(MemPoolAttr)>(
                           flags::MemPoolAttr::AllowInternalDependencies),
                         static_cast<void*>(&threshold));
}

GCXX_FH auto MemPoolView::SetReleaseThreshold(std::uint64_t threshold) -> void {
  GCXX_SAFE_RUNTIME_CALL(MemPoolSetAttribute,
                         "Failed to set ReleaseThreshold of mempool", pool_,
                         static_cast<GCXX_RUNTIME_BACKEND(MemPoolAttr)>(
                           flags::MemPoolAttr::ReleaseThreshold),
                         static_cast<void*>(&threshold));
}

GCXX_FH auto MemPoolView::SetReservedMemCurrent(std::uint64_t threshold)
  -> void {
  GCXX_SAFE_RUNTIME_CALL(MemPoolSetAttribute,
                         "Failed to set ReservedMemCurrent of mempool", pool_,
                         static_cast<GCXX_RUNTIME_BACKEND(MemPoolAttr)>(
                           flags::MemPoolAttr::ReservedMemCurrent),
                         static_cast<void*>(&threshold));
}

GCXX_FH auto MemPoolView::SetReservedMemHigh(std::uint64_t threshold) -> void {
  GCXX_SAFE_RUNTIME_CALL(MemPoolSetAttribute,
                         "Failed to set ReservedMemHigh of mempool", pool_,
                         static_cast<GCXX_RUNTIME_BACKEND(MemPoolAttr)>(
                           flags::MemPoolAttr::ReservedMemHigh),
                         static_cast<void*>(&threshold));
}

GCXX_FH auto MemPoolView::SetUsedMemCurrent(std::uint64_t threshold) -> void {
  GCXX_SAFE_RUNTIME_CALL(MemPoolSetAttribute,
                         "Failed to set UsedMemCurrent of mempool", pool_,
                         static_cast<GCXX_RUNTIME_BACKEND(MemPoolAttr)>(
                           flags::MemPoolAttr::UsedMemCurrent),
                         static_cast<void*>(&threshold));
}

GCXX_FH auto MemPoolView::SetUsedMemHigh(std::uint64_t threshold) -> void {
  GCXX_SAFE_RUNTIME_CALL(MemPoolSetAttribute,
                         "Failed to set UsedMemHigh of mempool", pool_,
                         static_cast<GCXX_RUNTIME_BACKEND(MemPoolAttr)>(
                           flags::MemPoolAttr::UsedMemHigh),
                         static_cast<void*>(&threshold));
}

GCXX_FH auto MemPoolView::GetFollowEventDependencies() -> bool {
  int retval{};
  GCXX_SAFE_RUNTIME_CALL(MemPoolGetAttribute,
                         "Failed to get FollowEventDependencies of mempool",
                         pool_,
                         static_cast<GCXX_RUNTIME_BACKEND(MemPoolAttr)>(
                           flags::MemPoolAttr::FollowEventDependencies),
                         static_cast<void*>(&retval));
  return static_cast<bool>(retval);
}

GCXX_FH auto MemPoolView::GetAllowOpportunistic() -> bool {
  int retval{};
  GCXX_SAFE_RUNTIME_CALL(MemPoolGetAttribute,
                         "Failed to get AllowOpportunistic of mempool", pool_,
                         static_cast<GCXX_RUNTIME_BACKEND(MemPoolAttr)>(
                           flags::MemPoolAttr::AllowOpportunistic),
                         static_cast<void*>(&retval));
  return static_cast<bool>(retval);
}

GCXX_FH auto MemPoolView::GetAllowInternalDependencies() -> bool {
  int retval{};
  GCXX_SAFE_RUNTIME_CALL(MemPoolGetAttribute,
                         "Failed to get AllowInternalDependencies of mempool",
                         pool_,
                         static_cast<GCXX_RUNTIME_BACKEND(MemPoolAttr)>(
                           flags::MemPoolAttr::AllowInternalDependencies),
                         static_cast<void*>(&retval));
  return static_cast<bool>(retval);
}

GCXX_FH auto MemPoolView::GetReleaseThreshold() -> std::uint64_t {
  std::uint64_t retval{};
  GCXX_SAFE_RUNTIME_CALL(MemPoolGetAttribute,
                         "Failed to get ReleaseThreshold of mempool", pool_,
                         static_cast<GCXX_RUNTIME_BACKEND(MemPoolAttr)>(
                           flags::MemPoolAttr::ReleaseThreshold),
                         static_cast<void*>(&retval));
  return retval;
}

GCXX_FH auto MemPoolView::GetReservedMemCurrent() -> std::uint64_t {
  std::uint64_t retval{};
  GCXX_SAFE_RUNTIME_CALL(MemPoolGetAttribute,
                         "Failed to get ReservedMemCurrent of mempool", pool_,
                         static_cast<GCXX_RUNTIME_BACKEND(MemPoolAttr)>(
                           flags::MemPoolAttr::ReservedMemCurrent),
                         static_cast<void*>(&retval));
  return retval;
}

GCXX_FH auto MemPoolView::GetReservedMemHigh() -> std::uint64_t {
  std::uint64_t retval{};
  GCXX_SAFE_RUNTIME_CALL(MemPoolGetAttribute,
                         "Failed to get ReservedMemHigh of mempool", pool_,
                         static_cast<GCXX_RUNTIME_BACKEND(MemPoolAttr)>(
                           flags::MemPoolAttr::ReservedMemHigh),
                         static_cast<void*>(&retval));
  return retval;
}

GCXX_FH auto MemPoolView::GetUsedMemCurrent() -> std::uint64_t {
  std::uint64_t retval{};
  GCXX_SAFE_RUNTIME_CALL(MemPoolGetAttribute,
                         "Failed to get UsedMemCurrent of mempool", pool_,
                         static_cast<GCXX_RUNTIME_BACKEND(MemPoolAttr)>(
                           flags::MemPoolAttr::UsedMemCurrent),
                         static_cast<void*>(&retval));
  return retval;
}

GCXX_FH auto MemPoolView::GetUsedMemHigh() -> std::uint64_t {
  std::uint64_t retval{};
  GCXX_SAFE_RUNTIME_CALL(MemPoolGetAttribute,
                         "Failed to get UsedMemHigh of mempool", pool_,
                         static_cast<GCXX_RUNTIME_BACKEND(MemPoolAttr)>(
                           flags::MemPoolAttr::UsedMemHigh),
                         static_cast<void*>(&retval));
  return retval;
}

GCXX_NAMESPACE_MAIN_END


#endif