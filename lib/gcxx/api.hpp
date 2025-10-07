#pragma once
#ifndef GCXX_API_HPP
#define GCXX_API_HPP

#include <gcxx/backend/backend.hpp>
// Section for different error handlings
// for example runtime, blas, etc.
#include <gcxx/runtime/runtime_error.hpp>

// Section for Runtime Flags
#include <gcxx/runtime/flags/eventflags.hpp>
#include <gcxx/runtime/flags/streamflags.hpp>


// Section for Runtime API
// #include <gcxx/runtime/device.hpp>
#include <gcxx/runtime/event.hpp>
#include <gcxx/runtime/stream.hpp>


// section for memory API
#include <gcxx/runtime/memory.hpp>


#endif