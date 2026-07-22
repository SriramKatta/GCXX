// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_HPP_
#define GCXX_RUNTIME_MEMORY_HPP_

#include <gcxx/runtime/memory/spans/spans.hpp>

// TODO: complete this at a later point
#include <gcxx/runtime/memory/mempool/mempool.hpp>
#include <gcxx/runtime/memory/mempool/mempool_view.hpp>

// CCCL-parity memory_pool module (device/managed/pinned pools + refs).
#include <gcxx/runtime/memory/memory_pool/device_memory_pool.hpp>
#include <gcxx/runtime/memory/memory_pool/managed_memory_pool.hpp>
#include <gcxx/runtime/memory/memory_pool/pinned_memory_pool.hpp>
#include <gcxx/runtime/memory/memory_pool/default_memory_pools.hpp>

#include <gcxx/runtime/memory/smartpointers/pointers.hpp>

#include <gcxx/runtime/memory/buffers/buffer.hpp>
#include <gcxx/runtime/memory/memory_resource/any_resource.hpp>
#include <gcxx/runtime/memory/memory_resource/resource_concepts.hpp>
#include <gcxx/runtime/memory/memory_resource/pooled_device_resource.hpp>
#include <gcxx/runtime/memory/memory_resource/resources.hpp>


#include <gcxx/runtime/memory/copy.hpp>
#include <gcxx/runtime/memory/fill.hpp>
#include <gcxx/runtime/memory/memory_helpers.hpp>
#include <gcxx/runtime/memory/memset.hpp>


#endif