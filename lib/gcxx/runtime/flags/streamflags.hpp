#pragma once
#ifndef GCXX_RUNTIME_FLAGS_STREAMFLAGS_HPP_
#define GCXX_RUNTIME_FLAGS_STREAMFLAGS_HPP_

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>

#include <limits>

GCXX_NAMESPACE_MAIN_BEGIN

namespace flags {

  /**
   * @brief Flags for selecting the type of GPU stream.
   *
   * These values control how a stream interacts with the special *NULL stream*
   * (a.k.a. "default stream", stream 0) provided by most GPU backends such as
   * CUDA and HIP.
   *
   * In CUDA:
   * - The NULL stream executes all operations sequentially with respect to
   *   *all* other streams that use `syncWithNull`.
   * - Streams created with `noSyncWithNull` are independent and may run
   *   concurrently with both the NULL stream and each other, subject to device
   *   resources.
   * - The `nullStream` flag provides explicit access to stream 0 itself.
   *
   * These flags are useful when tuning execution overlap and synchronization
   * behavior for kernels, memory copies, and host-device interactions.
   */
  enum class streamType : flag_t {
    /**
     * @brief Stream that synchronizes with the NULL stream.
     *
     * This is the *default* behavior on most backends: operations in this stream
     * will serialize with operations in stream 0, ensuring ordering but limiting
     * concurrency.
     */
    syncWithNull = GCXX_RUNTIME_BACKEND(StreamDefault),

    /**
     * @brief Stream that does not synchronize with the NULL stream.
     *
     * Allows concurrent execution with stream 0 and other non-blocking streams,
     * enabling overlap of kernels and memory transfers when hardware resources
     * permit.
     */
    noSyncWithNull = GCXX_RUNTIME_BACKEND(StreamNonBlocking),

    /**
     * @brief Explicit handle to the NULL stream (stream 0).
     *
     * This special sentinel value directly identifies the backend’s implicit
     * default stream. All work enqueued here executes in program order and
     * typically synchronizes with most other streams unless `noSyncWithNull`
     * semantics are used.
     */
    nullStream = std::numeric_limits<flag_t>::max()
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
  enum class streamPriority : flag_t {
    /// No priority hint; runtime chooses the default scheduling behavior.
    none = 0U,

    /// Hint for the lowest available priority (background work).
    veryLow = 1U,

    /// Hint for a slightly reduced priority compared to default.
    low = 2U,

    /// Hint for elevated priority; work may preempt lower-priority streams.
    high = 3U,

    /// Hint for very high priority, just below critical tasks.
    veryHigh = 4U,

    /// Hint for the absolute highest available priority (latency critical).
    critical = 5U,
  };

}  // namespace flags

GCXX_NAMESPACE_MAIN_END

#endif
