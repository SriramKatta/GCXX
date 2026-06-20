// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#include "tests_common.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include <gcxx/api.hpp>

namespace {

  template <typename T>
  auto memset_expected_value(const int byte_value) -> T {
    T value{};
    std::memset(&value, byte_value, sizeof(T));
    return value;
  }

  template <typename T, std::size_t N>
  void expect_device_contents_equal(T* device_ptr, const int byte_value) {
    std::array<T, N> host_values{};
    gcxx::memory::Copy(host_values.data(), device_ptr, N);

    const T expected = memset_expected_value<T>(byte_value);
    for (const auto& actual : host_values) {
      EXPECT_EQ(actual, expected);
    }
  }

}  // namespace

TEST(MemsetTest, RawPointerOverloadIsCallable) {
  EXPECT_TRUE(
    (std::is_same_v<decltype(gcxx::memory::Memset(
                      std::declval<std::uint32_t*&>(), 0, std::size_t{4})),
                    void>));
}

TEST(MemsetTest, UniquePointerOverloadIsCallable) {
  using device_ptr = gcxx::memory::device_ptr<std::uint32_t>;

  EXPECT_TRUE(
    (std::is_same_v<decltype(gcxx::memory::Memset(std::declval<device_ptr&>(),
                                                  0, std::size_t{4})),
                    void>));
}

TEST(MemsetTest, SpanOverloadIsCallable) {
  EXPECT_TRUE((std::is_same_v<decltype(gcxx::memory::Memset(
                                std::declval<gcxx::span<std::uint32_t>&>(), 0)),
                              void>));
}

TEST(MemsetTest, DeviceBufferOverloadIsCallable) {
  using device_buf = gcxx::memory::device_buffer<std::uint32_t>;

  EXPECT_TRUE(
    (std::is_same_v<decltype(gcxx::memory::Memset(
                      std::declval<device_buf&>(), 0)),
                    void>));
}