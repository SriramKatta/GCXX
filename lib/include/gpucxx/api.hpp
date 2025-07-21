#pragma once
#ifndef GPUCXX_API_WRAPPERS_HPP
#define GPUCXX_API_WRAPPERS_HPP

#include <gpucxx/backend/backend.hpp>
// Section for diffrent error handlings
// for example runtime, blas, etc.
#include <gpucxx/error/runtime_error.hpp>

// Section for Runtime Flags
#include <gpucxx/runtime/flags/eventflags.hpp>
#include <gpucxx/runtime/flags/streamflags.hpp>


// Section for API
#include <gpucxx/runtime/event.hpp>
#include <gpucxx/runtime/stream.hpp>

#endif