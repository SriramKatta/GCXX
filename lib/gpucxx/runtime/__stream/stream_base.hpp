#pragma once
#ifndef GPUCXX_RUNTIME_STREAM_STREAM_REF_HPP_
#define GPUCXX_RUNTIME_STREAM_STREAM_REF_HPP_

#include <gpucxx/backend/backend.hpp>
#include <gpucxx/macros/define_macros.hpp>

class stream_ref {
 public:
  stream_ref()               = default;
  stream_ref(int)            = delete;
  stream_ref(std::nullptr_t) = delete;

  stream_ref(deviceStream_t __str) : stream_(__str) {}

  GPUCXX_FHD constexpr auto get() const noexcept -> deviceStream_t {
    return stream_;
  }

  GPUCXX_FH auto HasPendingWork() -> bool;

  GPUCXX_FH auto Synchronize() const -> void;

 protected:
  using deviceStream_t = GPUCXX_RUNTIME_BACKEND(Stream_t);

 protected:
  deviceStream_t stream_{reinterpret_cast<deviceStream_t>(~0ULL)};
};

#endif