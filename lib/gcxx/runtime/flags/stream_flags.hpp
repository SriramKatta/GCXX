// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_FLAGS_STREAM_FLAGS_HPP_
#define GCXX_RUNTIME_FLAGS_STREAM_FLAGS_HPP_

#include <gcxx/internal/prologue.hpp>

#include <limits>

GCXX_NAMESPACE_MAIN_FLAGS_BEGIN()

// Stream type relative to the backend NULL/default stream (see enumerators).
enum class streamType : details_::flag_t {
  /**
   * @brief Stream that synchronizes with the NULL stream.
   *
   * This is the *default* behavior on most backends: operations in this
   * stream will serialize with operations in stream 0/NULL stream, ensuring
   * ordering but limiting concurrency.
   */
  SyncWithNull = GCXX_RUNTIME_BACKEND(StreamDefault),

  /**
   * @brief Stream that does not synchronize with the NULL stream.
   *
   * Allows concurrent execution with stream 0 and other non-blocking streams,
   * enabling overlap of kernels and memory transfers when hardware resources
   * permit.
   */
  NoSyncWithNull = GCXX_RUNTIME_BACKEND(StreamNonBlocking),

  /**
   * @brief Explicit handle to the NULL stream (stream 0).
   *
   * This special sentinel value directly identifies the backend’s implicit
   * default stream. All work enqueued here executes in program order and
   * typically synchronizes with most other streams unless `NoSyncWithNull`
   * semantics are used.
   */
  NullStream = std::numeric_limits<details_::flag_t>::max()
};

// Relative priority hints; the runtime maps them to the device's range.
enum class streamPriority : details_::flag_t {
  // No priority hint; runtime chooses the default scheduling behavior.
  None = 0U,

  // Hint for the lowest available priority (background work).
  VeryLow = 1U,

  // Hint for a slightly reduced priority compared to default.
  Low = 2U,

  // Hint for elevated priority; work may preempt lower-priority streams.
  High = 3U,

  // Hint for very High priority, just below Critical tasks.
  VeryHigh = 4U,

  // Hint for the absolute highest available priority (latency Critical).
  Critical = 5U,
};

// Scope of safety checks during graph capture (global/thread/relaxed).
enum class streamCaptureMode : details_::flag_t {
  Global = GCXX_RUNTIME_BACKEND(
    StreamCaptureModeGlobal), /**< Capture affects all threads; any unsafe API
                                 call results in an error. */
  ThreadLocal = GCXX_RUNTIME_BACKEND(
    StreamCaptureModeThreadLocal), /**< Capture is local to the thread; other
                                      threads can use unsafe APIs. */
  Relaxed = GCXX_RUNTIME_BACKEND(
    StreamCaptureModeRelaxed), /**< No safety checks; application must ensure
                                  correctness. */
};

enum class streamCaptureStatus : details_::flag_t {
  None        = GCXX_RUNTIME_BACKEND(StreamCaptureStatusNone),
  Active      = GCXX_RUNTIME_BACKEND(StreamCaptureStatusActive),
  Invalidated = GCXX_RUNTIME_BACKEND(StreamCaptureStatusInvalidated),
};

#if GCXX_CUDA_MODE()
inline streamCaptureStatus to_streamCaptureStatus(
  GCXX_RUNTIME_BACKEND(StreamCaptureStatus) status) {
  switch (status) {
    case cudaStreamCaptureStatusNone:
      return streamCaptureStatus::None;
    case cudaStreamCaptureStatusActive:
      return streamCaptureStatus::Active;
    case cudaStreamCaptureStatusInvalidated:
      return streamCaptureStatus::Invalidated;
    default:
      return streamCaptureStatus::None;  // or handle error
  }
}

enum class StreamUpdateCaptureDependencies : details_::flag_t {
  Add = GCXX_RUNTIME_BACKEND(StreamAddCaptureDependencies),
  Set = GCXX_RUNTIME_BACKEND(StreamSetCaptureDependencies),
};
#endif
GCXX_NAMESPACE_MAIN_FLAGS_END()

#endif
