#pragma once
#ifndef GCXX_API_HPP
#define GCXX_API_HPP
// clang-format off
#include <gcxx/prologue.hpp>

// section for types
#include <gcxx/types/shared_mem.hpp>
#include <gcxx/types/vector_types.hpp>

// Section for different error handlings
// for example runtime, blas, etc.
#include <gcxx/runtime/runtime_error.hpp>

// Section for Runtime Flags
#include <gcxx/runtime/flags/device_flags.hpp>
#include <gcxx/runtime/flags/event_flags.hpp>
#include <gcxx/runtime/flags/stream_flags.hpp>
#include <gcxx/runtime/flags/graph_flags.hpp>

// section for device handler
#include <gcxx/runtime/device.hpp>


// Section for Runtime API
#include <gcxx/runtime/device.hpp>
#include <gcxx/runtime/event.hpp>
#include <gcxx/runtime/stream.hpp>
#include <gcxx/runtime/graph.hpp>
#include <gcxx/runtime/launch.hpp>


// section for memory API
#include <gcxx/runtime/memory.hpp>

// clang-format on
#endif