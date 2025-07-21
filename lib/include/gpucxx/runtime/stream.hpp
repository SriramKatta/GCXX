#pragma once
#ifndef GPUCXX_API_RUNTIME_STREAM_HPP
#define GPUCXX_API_RUNTIME_STREAM_HPP

#include <gpucxx/backend/backend.hpp>
#include <gpucxx/error/runtime_error.hpp>
#include <gpucxx/runtime/flags/eventflags.hpp>
#include <gpucxx/runtime/flags/streamflags.hpp>
#include <gpucxx/utils/define_specifiers.hpp>

#include <utility>

GPUCXX_BEGIN_NAMESPACE

class Stream {
 private:
  using deviceRawEvent_t  = GPUCXX_RUNTIME_BACKEND(Event_t);
  using deviceRawStream_t = GPUCXX_RUNTIME_BACKEND(Stream_t);
  deviceRawStream_t stream_{nullptr};

  GPUCXX_FH Stream(const flags::streamCreate createFlag,
                   const flags::streamPriority priorityFlag) {
    if (createFlag == flags::streamCreate::Null) {
      return;
    }
    GPUCXX_SAFE_RUNTIME_CALL(StreamCreateWithPriority,
                             (&stream_, static_cast<flag_t>(createFlag),
                              -static_cast<flag_t>(priorityFlag)));
  }

  GPUCXX_FH auto destroy() -> void {
    if (stream_) {
      GPUCXX_SAFE_RUNTIME_CALL(StreamDestroy, (stream_));
      stream_ = nullptr;
    }
  }

 public:
  GPUCXX_FH friend auto StreamCreate(const flags::streamCreate,
                                     const flags::streamPriority) -> Stream;

  GPUCXX_FH ~Stream() { this->destroy(); }

  GPUCXX_FH auto query() const -> bool {
    auto err            = GPUCXX_RUNTIME_BACKEND(StreamQuery)(stream_);
    constexpr auto line = __LINE__ - 1;
    switch (err) {
      case GPUCXX_RUNTIME_BACKEND(Success):
        return true;
      case GPUCXX_RUNTIME_BACKEND(ErrorNotReady):
        return false;
      default:
        details_::checkDeviceError(err, "stream Query", __FILE__, line);
        return false;
    }
  }

  GPUCXX_FH auto Synchronize() const -> void {
    GPUCXX_SAFE_RUNTIME_CALL(StreamSynchronize, (stream_));
  }

  GPUCXX_FH auto WaitOnEvent(
    const deviceRawEvent_t &event,
    const flags::eventWait waitFlag = flags::eventWait::Default) const -> void {
    GPUCXX_SAFE_RUNTIME_CALL(StreamWaitEvent,
                             (stream_, event, static_cast<flag_t>(waitFlag)));
  }

  Stream(const Stream &)            = delete;
  Stream &operator=(const Stream &) = delete;
  Stream(nullptr_t)                 = delete;
  Stream(int)                       = delete;

  Stream(Stream &&other) noexcept
      : stream_(std::exchange(other.stream_, nullptr)) {}

  GPUCXX_FH operator deviceRawStream_t() const { return stream_; }

  GPUCXX_FH deviceRawStream_t release() noexcept {
    return std::exchange(stream_, nullptr);
  }

  GPUCXX_FH auto operator=(Stream &&other) noexcept -> Stream & {
    if (this != &other) {
      this->destroy();
      stream_ = std::exchange(other.stream_, nullptr);
    }
    return *this;
  }

  GPUCXX_FH auto getPriority() -> flags::streamPriority {
    int prio{0};
    GPUCXX_SAFE_RUNTIME_CALL(StreamGetPriority, (stream_, &prio));
    return static_cast<flags::streamPriority>(-prio);
  }
};

GPUCXX_FH auto StreamCreate(
  const flags::streamCreate createFlag     = flags::streamCreate::Default,
  const flags::streamPriority priorityFlag = flags::streamPriority::lowest)
  -> Stream {
  return {createFlag, priorityFlag};
}

GPUCXX_END_NAMESPACE

#endif