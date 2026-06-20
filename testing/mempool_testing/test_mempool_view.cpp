// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#include "tests_common.hpp"

#include <cstddef>
#include <cstdint>

#include <gcxx/api.hpp>

namespace {

  // Pooled allocation sizes used below; named to keep them out of the
  // readability-magic-numbers / bugprone-argument-comment crosshairs.
  constexpr std::size_t kSmallAllocBytes         = 1024;
  constexpr std::size_t kLargeAllocBytes         = 4096;
  constexpr std::uint64_t kReleaseThresholdBytes = 1U << 20;  // 1 MiB

  // Query the device count via the raw backend call (not the throwing
  // driver:: wrapper, which aborts when GCXX_WITH_EXCEPTIONS is off) so the
  // fixture can skip gracefully on hosts without a usable GPU driver.
  auto gpuAvailable() -> bool {
    int count      = 0;
    const auto err = ::GCXX_RUNTIME_BACKEND(GetDeviceCount)(&count);
    return err == gcxx::driver::deviceErrSuccess && count > 0;
  }

}  // namespace

class MemPoolViewTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!gpuAvailable()) {
      GTEST_SKIP() << "No GPU device available; skipping pool view tests";
    }
  }
};

TEST_F(MemPoolViewTest, GetDefaultMempoolViaStatic) {
  const auto dev               = gcxx::Device::get();
  const gcxx::MemPoolView view = gcxx::MemPoolView::GetDefaultMempool(dev);
  EXPECT_NE(view.getRawMemPool(), nullptr);
}

TEST_F(MemPoolViewTest, GetDefaultMempoolViaHandle) {
  const auto dev               = gcxx::Device::get();
  const gcxx::MemPoolView view = dev.GetDefaultMemPool();
  EXPECT_NE(view.getRawMemPool(), nullptr);
}

TEST_F(MemPoolViewTest, BooleanAttributeRoundTrip) {
  const gcxx::MemPool pool(gcxx::MemPoolProps{});
  gcxx::MemPoolView view = pool.get();

  view.SetFollowEventDependencies(false);
  EXPECT_FALSE(view.GetFollowEventDependencies());

  view.SetFollowEventDependencies(true);
  EXPECT_TRUE(view.GetFollowEventDependencies());
}

TEST_F(MemPoolViewTest, ReleaseThresholdRoundTrip) {
  const gcxx::MemPool pool(gcxx::MemPoolProps{});
  gcxx::MemPoolView view = pool.get();

  view.SetReleaseThreshold(kReleaseThresholdBytes);
  EXPECT_EQ(view.GetReleaseThreshold(), kReleaseThresholdBytes);

  view.SetReleaseThreshold(0U);
  EXPECT_EQ(view.GetReleaseThreshold(), 0U);
}

TEST_F(MemPoolViewTest, MallocFromPoolAsyncReturnsPointer) {
  const gcxx::MemPool pool(gcxx::MemPoolProps{});
  const gcxx::MemPoolView view = pool.get();

  const gcxx::StreamView stream = gcxx::StreamView::Null();
  void* const ptr = view.MallocFromPoolAsync(kSmallAllocBytes, stream);
  EXPECT_NE(ptr, nullptr);

  gcxx::driver::deviceFreeAsync(ptr, stream.getRawStream());
}

TEST_F(MemPoolViewTest, TrimToIsCallable) {
  const gcxx::MemPool pool(gcxx::MemPoolProps{});
  const gcxx::MemPoolView view = pool.get();

  // Allocate then trim; keeping 0 bytes releases all idle memory back to OS.
  const gcxx::StreamView stream = gcxx::StreamView::Null();
  void* const ptr = view.MallocFromPoolAsync(kLargeAllocBytes, stream);
  ASSERT_NE(ptr, nullptr);
  gcxx::driver::deviceFreeAsync(ptr, stream.getRawStream());

  view.TrimTo(0U);
  SUCCEED();
}

TEST_F(MemPoolViewTest, GetAccessForLocalDeviceIsReadWrite) {
  const gcxx::MemPool pool(gcxx::MemPoolProps{});
  gcxx::MemPoolView view = pool.get();

  gcxx::MemAccessDesc loc;
  loc.locationType = gcxx::flags::MemLocation::Device;
  loc.locationId   = gcxx::driver::deviceGet();
  loc.flags        = gcxx::flags::MemAccessFlags::ReadWrite;

  view.SetAccess(loc);
  EXPECT_EQ(view.GetAccess(loc), gcxx::flags::MemAccessFlags::ReadWrite);
}
