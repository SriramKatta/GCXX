// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_BACKEND_BACKEND_HANDLES_HPP_
#define GCXX_RUNTIME_BACKEND_BACKEND_HANDLES_HPP_

#include <gcxx/internal/prologue.hpp>

GCXX_NAMESPACE_MAIN_DRIVER_BEGIN

// ╔════════════════════════════════════════════════════════╗
// ║                     Error handles                      ║
// ╚════════════════════════════════════════════════════════╝
using deviceError_t                     = GCXX_RUNTIME_BACKEND(Error_t);
GCXX_CXPR inline auto deviceErrSuccess  = GCXX_RUNTIME_BACKEND(Success);
GCXX_CXPR inline auto deviceErrNotReady = GCXX_RUNTIME_BACKEND(ErrorNotReady);

// ╔════════════════════════════════════════════════════════╗
// ║                     Sream Handles                      ║
// ╚════════════════════════════════════════════════════════╝
using deviceStream_t = GCXX_RUNTIME_BACKEND(Stream_t);
inline static constexpr deviceStream_t NULL_STREAM{nullptr};  // NOLINT

// clang-format off
inline static const auto INVALID_STREAM = reinterpret_cast<deviceStream_t>(~0ULL);  // NOLINT
// clang-format on


// ╔════════════════════════════════════════════════════════╗
// ║                     Event handles                      ║
// ╚════════════════════════════════════════════════════════╝
using deviceEvent_t = GCXX_RUNTIME_BACKEND(Event_t);
inline static constexpr deviceEvent_t INVALID_EVENT{nullptr};  // NOLINT


// ╔════════════════════════════════════════════════════════╗
// ║                     Graph handles                      ║
// ╚════════════════════════════════════════════════════════╝
using deviceGraph_t     = GCXX_RUNTIME_BACKEND(Graph_t);
using deviceGraphNode_t = GCXX_RUNTIME_BACKEND(GraphNode_t);
using deviceGraphExec_t = GCXX_RUNTIME_BACKEND(GraphExec_t);


GCXX_NAMESPACE_MAIN_DRIVER_END

#endif