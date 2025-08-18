#pragma once
#ifndef GPUCXX_API_HPP
#define GPUCXX_API_HPP

#include <gpucxx/backend/backend.hpp>
// Section for different error handlings
// for example runtime, blas, etc.
#include <gpucxx/runtime/runtime_error.hpp>

// Section for Runtime Flags
#include <gpucxx/runtime/__flags/eventflags.hpp>
#include <gpucxx/runtime/__flags/streamflags.hpp>


// Section for Runtime API
//#include <gpucxx/runtime/device.hpp>
#include <gpucxx/runtime/stream.hpp>
#include <gpucxx/runtime/event.hpp>


#endif