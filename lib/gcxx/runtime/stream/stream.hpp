#pragma once
#ifndef GCXX_API_RUNTIME_STREAM_STREAM_HPP_
#define GCXX_API_RUNTIME_STREAM_STREAM_HPP_

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>
#include <gcxx/runtime/flags/eventflags.hpp>
#include <gcxx/runtime/flags/streamflags.hpp>
#include <gcxx/runtime/stream/stream_wrap.hpp>

#include <cstddef>
#include <utility>

GCXX_NAMESPACE_MAIN_BEGIN

class Stream : public stream_wrap {
 public:
  GCXX_FH Stream(
    const flags::streamType createFlag       = flags::streamType::syncWithNull,
    const flags::streamPriority priorityFlag = flags::streamPriority::none);

  GCXX_FH static auto Create(
    const flags::streamType createFlag       = flags::streamType::syncWithNull,
    const flags::streamPriority priorityFlag = flags::streamPriority::none)
    -> Stream;

  GCXX_FH ~Stream();

  Stream(int) = delete;

  Stream(std::nullptr_t) = delete;

  Stream(const Stream&) = delete;

  Stream& operator=(const Stream&) = delete;

  GCXX_FH Stream(Stream&& other) noexcept
      : stream_wrap(std::exchange(other.stream_, details_::INVALID_STREAM)) {}

  GCXX_FH auto operator=(Stream&& other) GCXX_NOEXCEPT->Stream&;


  GCXX_FH auto release() GCXX_NOEXCEPT -> stream_wrap;


  GCXX_FH auto getPriority() -> flags::streamPriority;

 private:
  GCXX_FH auto destroy() -> void;
};

GCXX_NAMESPACE_MAIN_END

#include <gcxx/runtime/details/stream.inl>

#include <gcxx/macros/undefine_macros.hpp>

#endif
