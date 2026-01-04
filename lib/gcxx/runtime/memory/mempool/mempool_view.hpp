#pragma once
#ifndef GCXX_RUNTIME_MEMEORY_MEMPOOL_MEMPOOL_VIEW_HPP_
#define GCXX_RUNTIME_MEMEORY_MEMPOOL_MEMPOOL_VIEW_HPP_

#include <gcxx/internal/prologue.hpp>

#include <cstdint>

#include <gcxx/runtime/device/device_handle.hpp>
#include <gcxx/runtime/flags/memory_flags.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

using deviceMemPool_t = GCXX_RUNTIME_BACKEND(MemPool_t);

class MemPoolView {
 protected:
  deviceMemPool_t pool_{nullptr};

 public:
  MemPoolView() = default;

  MemPoolView(deviceMemPool_t pool) : pool_(pool) {}

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
};

GCXX_NAMESPACE_MAIN_END


#include <gcxx/runtime/details/memory/mempool/mempool_view.inl>

#endif