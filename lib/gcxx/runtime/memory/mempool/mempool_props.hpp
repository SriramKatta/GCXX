// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_MEMPOOL_MEMPOOL_PROPS_HPP_
#define GCXX_RUNTIME_MEMORY_MEMPOOL_MEMPOOL_PROPS_HPP_

#include <cstddef>
#include <cstring>

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/flags/memory_flags.hpp>
#include <gcxx/runtime_backend/backend_handles.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

using deviceMemPoolProps_t = driver::deviceMemPoolProps_t;

struct MemPoolProps {
  flags::MemAllocation allocType{flags::MemAllocation::Pinned};
  details_::flag_t handleTypes{flags::MemAllocationHandle::None};
  flags::MemLocation locationType{flags::MemLocation::Device};
  int locationId{0};
  void* win32SecurityAttributes{nullptr};
#if GCXX_CUDA_MODE()
  std::size_t maxSize{0};
  unsigned short usage{0};
#endif

  GCXX_FH auto getRawMemPoolProps() const -> deviceMemPoolProps_t {
    deviceMemPoolProps_t props{};
    std::memset(&props, 0, sizeof(props));
    props.allocType = static_cast<driver::deviceMemAllocationType_t>(allocType);
    props.handleTypes =
      static_cast<driver::deviceMemAllocationHandleType_t>(handleTypes);
    props.location.type =
      static_cast<decltype(props.location.type)>(locationType);
    props.location.id             = locationId;
    props.win32SecurityAttributes = win32SecurityAttributes;
#if GCXX_CUDA_MODE()
    props.maxSize = maxSize;
    props.usage   = usage;
#endif
    return props;
  }
};

// Pool peer-access descriptor (used by MemPoolView's access API).
struct MemAccessDesc {
  flags::MemLocation locationType{flags::MemLocation::Device};
  int locationId{0};
  flags::MemAccessFlags flags{flags::MemAccessFlags::ReadWrite};

  GCXX_FH auto getRawMemAccessDesc() const -> driver::deviceMemAccessDesc_t {
    driver::deviceMemAccessDesc_t desc{};
    desc.location.type =
      static_cast<decltype(desc.location.type)>(locationType);
    desc.location.id = locationId;
    desc.flags       = static_cast<driver::deviceMemAccessFlags_t>(flags);
    return desc;
  }

  GCXX_FH auto getRawMemLocation() const -> driver::deviceMemLocation_t {
    driver::deviceMemLocation_t loc{};
    loc.type = static_cast<decltype(loc.type)>(locationType);
    loc.id   = locationId;
    return loc;
  }
};

GCXX_NAMESPACE_MAIN_END()


#endif