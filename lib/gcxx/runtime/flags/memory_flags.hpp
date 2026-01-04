#pragma once
#ifndef GCXX_RUNTIME_FLAGS_MEMORY_FLAGS_HPP_
#define GCXX_RUNTIME_FLAGS_MEMORY_FLAGS_HPP_

#include <gcxx/internal/prologue.hpp>

#include <limits>

GCXX_NAMESPACE_MAIN_FLAGS_BEGIN

enum class MemAllocation : details_::flag_t {
  Invalid = GCXX_RUNTIME_BACKEND(MemAllocationTypeInvalid),
  Pinned  = GCXX_RUNTIME_BACKEND(MemAllocationTypePinned),
  Max     = GCXX_RUNTIME_BACKEND(MemAllocationTypeMax),
};

enum MemAllocationHandle : details_::flag_t {
  None                = GCXX_RUNTIME_BACKEND(MemHandleTypeNone),
  PosixFileDescriptor = GCXX_RUNTIME_BACKEND(MemHandleTypePosixFileDescriptor),
  Win32               = GCXX_RUNTIME_BACKEND(MemHandleTypeWin32),
  Win32Kmt            = GCXX_RUNTIME_BACKEND(MemHandleTypeWin32Kmt),
  Fabric              = GCXX_RUNTIME_BACKEND(MemHandleTypeFabric),
};

enum class MemLocation : details_::flag_t {
  Invalid         = GCXX_RUNTIME_BACKEND(MemLocationTypeInvalid),
  Device          = GCXX_RUNTIME_BACKEND(MemLocationTypeDevice),
  Host            = GCXX_RUNTIME_BACKEND(MemLocationTypeHost),
  HostNuma        = GCXX_RUNTIME_BACKEND(MemLocationTypeHostNuma),
  HostNumaCurrent = GCXX_RUNTIME_BACKEND(MemLocationTypeHostNumaCurrent),
};


enum class MemPoolAttr : details_::flag_t {
  FollowEventDependencies =
    GCXX_RUNTIME_BACKEND(MemPoolReuseFollowEventDependencies),
  AllowOpportunistic = GCXX_RUNTIME_BACKEND(MemPoolReuseAllowOpportunistic),
  AllowInternalDependencies =
    GCXX_RUNTIME_BACKEND(MemPoolReuseAllowInternalDependencies),
  ReleaseThreshold   = GCXX_RUNTIME_BACKEND(MemPoolAttrReleaseThreshold),
  ReservedMemCurrent = GCXX_RUNTIME_BACKEND(MemPoolAttrReservedMemCurrent),
  ReservedMemHigh    = GCXX_RUNTIME_BACKEND(MemPoolAttrReservedMemHigh),
  UsedMemCurrent     = GCXX_RUNTIME_BACKEND(MemPoolAttrUsedMemCurrent),
  UsedMemHigh        = GCXX_RUNTIME_BACKEND(MemPoolAttrUsedMemHigh),
};


GCXX_NAMESPACE_MAIN_FLAGS_END

#endif
