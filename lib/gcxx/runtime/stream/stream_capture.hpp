// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_STREAM_STREAM_CAPTURE_HPP_
#define GCXX_RUNTIME_STREAM_STREAM_CAPTURE_HPP_

// Stream graph-capture support, split out of stream_view so that TUs which only
// need streams/events do not transitively parse the entire graph subsystem.
#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>

#include <gcxx/runtime/details/stream/stream_capture.inl>

#endif
