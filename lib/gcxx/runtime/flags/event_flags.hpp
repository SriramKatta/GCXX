/**
 * @file event_flags.hpp
 * @brief Event-related flags for GPU runtime operations.
 *
 * This header defines enumerations for controlling event creation, recording,
 * waiting, and stream capture behavior in a backend-agnostic manner (CUDA/HIP).
 */

#pragma once
#ifndef GCXX_RUNTIME_FLAGS_EVENT_FLAGS_HPP_
#define GCXX_RUNTIME_FLAGS_EVENT_FLAGS_HPP_

#include <gcxx/internal/prologue.hpp>

GCXX_NAMESPACE_MAIN_FLAGS_BEGIN

/**
 * @enum eventCreate
 * @brief Flags for controlling GPU event creation behavior.
 *
 * These flags specify how an event should be created, including synchronization
 * behavior, timing capabilities, and inter-process sharing.
 */
enum class eventCreate : details_::flag_t {
  None = GCXX_RUNTIME_BACKEND(EventDefault), /**< Default event creation. */
  blockingSync =
    GCXX_RUNTIME_BACKEND(EventBlockingSync), /**< CPU thread blocks on
                                                synchronization. */
  disableTiming =
    GCXX_RUNTIME_BACKEND(EventDisableTiming), /**< Disable timing to improve
                                                 performance. */
  interprocess =
    GCXX_RUNTIME_BACKEND(EventInterprocess) |
    GCXX_RUNTIME_BACKEND(EventDisableTiming), /**< Enable inter-process sharing
                                                 (timing disabled). */
};

/**
 * @brief Bitwise OR operator for combining eventCreate flags.
 * @param lhs Left-hand side eventCreate flag.
 * @param rhs Right-hand side eventCreate flag.
 * @return Combined eventCreate flags.
 */
inline auto operator|(const eventCreate& lhs, const eventCreate& rhs)
  -> eventCreate {
  return static_cast<eventCreate>(static_cast<details_::flag_t>(lhs) |
                                  static_cast<details_::flag_t>(rhs));
}


/**
 * @enum eventRecord
 * @brief Flags for controlling event recording behavior.
 *
 * These flags specify how an event should be recorded in a stream.
 */
enum class eventRecord : details_::flag_t {
  None = GCXX_RUNTIME_BACKEND(EventRecordDefault),      /**< Default recording
                                                           behavior. */
  external = GCXX_RUNTIME_BACKEND(EventRecordExternal), /**< Record for external
                                                           synchronization. */
};

/**
 * @enum eventWait
 * @brief Flags for controlling event wait behavior.
 *
 * These flags specify how a stream should wait on an event.
 *
 * @note HIP does not currently implement these flags despite documentation
 *       claiming support. For HIP, both values default to 0.
 */
enum class eventWait : details_::flag_t {
#if defined(GCXX_CUDA_MODE)
  None = GCXX_RUNTIME_BACKEND(EventWaitDefault), /**< Default wait behavior. */
  external =
    GCXX_RUNTIME_BACKEND(EventWaitExternal), /**< Wait on external event. */
#else  // HIP: these are supposedly defined as per documentation but not
       // implemented in the actual code
  None     = 0, /**< Default wait behavior (HIP fallback). */
  external = 0, /**< External wait (HIP fallback, not implemented). */
#endif
};

GCXX_NAMESPACE_MAIN_FLAGS_END

#endif