// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_API_HPP
#define GCXX_API_HPP
// clang-format off
#include <gcxx/internal/prologue.hpp>



// make spans available by default without pulling runtime
#include <gcxx/spans.hpp>

// heavy runtime API (GPU-related) is included only when not in CPU-only mode
#if !GCXX_CPU_MODE()

// section for types
#include <gcxx/types/shared_mem.hpp>
#include <gcxx/types/vector_types.hpp>
#include <gcxx/types/vector_types_op.hpp>

// Section for Runtime Flags
#include <gcxx/runtime/flags/device_flags.hpp>
#include <gcxx/runtime/flags/memory_flags.hpp>
#include <gcxx/runtime/flags/event_flags.hpp>
#include <gcxx/runtime/flags/stream_flags.hpp>
#include <gcxx/runtime/flags/graph_flags.hpp>
#include <gcxx/runtime/flags/occupancy_flags.hpp>

// Section for Runtime API
#include <gcxx/runtime/device.hpp>
#include <gcxx/runtime/event.hpp>
#include <gcxx/runtime/stream.hpp>
#include <gcxx/runtime/graph.hpp>
#include <gcxx/runtime/launch.hpp>
#include <gcxx/runtime/occupancy.hpp>


// section for memory API
#include <gcxx/runtime/memory.hpp>
#endif

// clang-format on
#endif