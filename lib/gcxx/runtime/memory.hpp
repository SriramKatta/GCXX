// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_HPP_
#define GCXX_RUNTIME_MEMORY_HPP_

#include <gcxx/runtime/memory/spans/spans.hpp>


#include <gcxx/runtime/memory/mempool/mempool.hpp>

#include <gcxx/runtime/memory/mempool/default_memory_pools.hpp>
#include <gcxx/runtime/memory/mempool/device_memory_pool.hpp>
#include <gcxx/runtime/memory/mempool/managed_memory_pool.hpp>
#include <gcxx/runtime/memory/mempool/pinned_memory_pool.hpp>

#include <gcxx/runtime/memory/smartpointers/pointers.hpp>

#include <gcxx/runtime/memory/buffers/buffer.hpp>
#include <gcxx/runtime/memory/memory_resource/any_resource.hpp>
#include <gcxx/runtime/memory/memory_resource/resource_concepts.hpp>


#include <gcxx/runtime/memory/copy.hpp>
#include <gcxx/runtime/memory/fill.hpp>
#include <gcxx/runtime/memory/memory_helpers.hpp>
#include <gcxx/runtime/memory/memset.hpp>


#endif