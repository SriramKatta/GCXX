#pragma once
#ifndef GCXX_RUNTIME_FLAGS_STREAM_FLAGS_HPP_
#define GCXX_RUNTIME_FLAGS_STREAM_FLAGS_HPP_

#include <gcxx/internal/prologue.hpp>

#include <limits>

GCXX_NAMESPACE_MAIN_FLAGS_BEGIN

/**
 * @brief Flags for selecting the type of GPU stream.
 *
 * These values control how a stream interacts with the special *NULL stream*
 * (a.k.a. "default stream", stream 0) provided by most GPU backends such as
 * CUDA and HIP.
 *
 * In CUDA:
 * - The NULL stream executes all operations sequentially with respect to
 *   *all* other streams that use `SyncWithNull`.
 * - Streams created with `NoSyncWithNull` are independent and may run
 *   concurrently with both the NULL stream and each other, subject to device
 *   resources.
 * - The `NullStream` flag provides explicit access to stream 0 itself.
 *
 * These flags are useful when tuning execution overlap and synchronization
 * behavior for kernels, memory copies, and host-device interactions.
 */
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

/**
 * @brief Flags for specifying scheduling priority of GPU streams.
 *
 * Many GPU backends (CUDA, HIP) allow streams to be created with different
 * priorities. Higher-priority streams may be scheduled before lower-priority
 * streams when the device is oversubscribed. Priority can influence latency-
 * sensitive workloads (e.g. interactive tasks) versus throughput workloads
 * (e.g. large batch kernels).
 *
 * Note:
 * - Not all devices or backends support fine-grained priority ranges.
 * - The values defined here are relative hints; the runtime maps them to the
 *   actual priority range supported by the device.
 */
enum class streamPriority : details_::flag_t {
  /// No priority hint; runtime chooses the default scheduling behavior.
  None = 0U,

  /// Hint for the lowest available priority (background work).
  VeryLow = 1U,

  /// Hint for a slightly reduced priority compared to default.
  Low = 2U,

  /// Hint for elevated priority; work may preempt lower-priority streams.
  High = 3U,

  /// Hint for very High priority, just below Critical tasks.
  VeryHigh = 4U,

  /// Hint for the absolute highest available priority (latency Critical).
  Critical = 5U,
};

/**
 * @enum streamCaptureMode
 * @brief Flags for controlling stream capture mode behavior.
 *
 * These flags determine how stream capture interacts with other streams
 * and threads during CUDA/HIP graph capture operations.
 */
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

#if GCXX_CUDA_MODE
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
GCXX_NAMESPACE_MAIN_FLAGS_END

#endif
