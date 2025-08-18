#pragma once
#ifndef GPUCXX_API_RUNTIME_STREAM_STREAM_HPP_
#define GPUCXX_API_RUNTIME_STREAM_STREAM_HPP_

#include <gpucxx/backend/backend.hpp>
#include <gpucxx/macros/define_macros.hpp>
#include <gpucxx/runtime/__flags/eventflags.hpp>
#include <gpucxx/runtime/__flags/streamflags.hpp>

#include <cstddef>
#include <utility>

GPUCXX_BEGIN_NAMESPACE

class Stream : stream_ref {
 private:
  GPUCXX_FH Stream(const flags::streamBehaviour createFlag,
                   const flags::streamPriority priorityFlag);

  GPUCXX_FH auto destroy() -> void;

 public:
  GPUCXX_FH static auto Create(
    const flags::streamBehaviour createFlag  = flags::streamBehaviour::none,
    const flags::streamPriority priorityFlag = flags::streamPriority::none)
    -> Stream;

  GPUCXX_FH ~Stream();

  Stream(int)                      = delete;
  Stream(std::nullptr_t)           = delete;
  Stream(const Stream&)            = delete;
  Stream& operator=(const Stream&) = delete;

  Stream(Stream&& other) noexcept;

  GPUCXX_FH operator deviceStream_t() const;

  GPUCXX_FH deviceStream_t release() noexcept;
  GPUCXX_FH auto operator=(Stream&& other) noexcept -> Stream&;

  GPUCXX_FH auto getPriority() -> flags::streamPriority;
};

GPUCXX_END_NAMESPACE

#include <gpucxx/runtime/__details/stream.inl>

#include <gpucxx/macros/undefine_macros.hpp>

#endif