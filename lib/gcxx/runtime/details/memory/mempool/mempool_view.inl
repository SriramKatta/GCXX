// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_MEMEORY_MEMPOOL_MEMPOOL_VIEW_INL_
#define GCXX_RUNTIME_DETAILS_MEMEORY_MEMPOOL_MEMPOOL_VIEW_INL_

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/device/device_handle.hpp>
#include <gcxx/runtime/flags/memory_flags.hpp>
#include <gcxx/runtime_backend/backend_stream_memory.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_FH auto MemPoolView::getRawMemPool() const -> deviceMemPool_t {
  return m_pool;
};

GCXX_FH auto MemPoolView::GetDefaultMempool(const DeviceHandle& hand)
  -> MemPoolView {
  return hand.GetDefaultMemPool();
}

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