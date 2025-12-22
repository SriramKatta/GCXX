#pragma once
#ifndef GCXX_API_RUNTIME_STREAM_STREAM_HPP_
#define GCXX_API_RUNTIME_STREAM_STREAM_HPP_

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>
#include <gcxx/runtime/flags/event_flags.hpp>
#include <gcxx/runtime/flags/stream_flags.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>

#include <cstddef>
#include <utility>

GCXX_NAMESPACE_MAIN_BEGIN

class Stream : public StreamView {
 private:
  GCXX_FH auto destroy() -> void;

 public:
  GCXX_FH Stream(
    const flags::streamType createFlag       = flags::streamType::SyncWithNull,
    const flags::streamPriority priorityFlag = flags::streamPriority::None);

  GCXX_FH static auto Create(
    const flags::streamType createFlag       = flags::streamType::SyncWithNull,
    const flags::streamPriority priorityFlag = flags::streamPriority::None)
    -> Stream;

  GCXX_FH ~Stream();

  Stream(int) = delete;

  Stream(std::nullptr_t) = delete;

  Stream(const Stream&) = delete;

  Stream& operator=(const Stream&) = delete;

  GCXX_FH Stream(Stream&& other) noexcept;

  GCXX_FH auto operator=(Stream&& other) GCXX_NOEXCEPT->Stream&;

  GCXX_FH constexpr auto get() GCXX_CONST_NOEXCEPT->StreamView;

  GCXX_FH auto Release() GCXX_NOEXCEPT->StreamView;

  GCXX_FH auto getPriority() -> flags::streamPriority;
};

GCXX_NAMESPACE_MAIN_END

#include <gcxx/runtime/details/stream/stream.inl>

#include <gcxx/macros/undefine_macros.hpp>

#endif
