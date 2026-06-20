// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#include "tests_common.hpp"

#include <utility>

#include <gcxx/api.hpp>

namespace {

  auto makeProps() -> gcxx::MemPoolProps {
    gcxx::MemPoolProps props;
    props.locationId = gcxx::driver::deviceGet();
    return props;
  }

  // Query the device count via the raw backend call (not the throwing
  // driver:: wrapper, which aborts when GCXX_WITH_EXCEPTIONS is off) so the
  // fixture can skip gracefully on hosts without a usable GPU driver.
  auto gpuAvailable() -> bool {
    int count      = 0;
    const auto err = ::GCXX_RUNTIME_BACKEND(GetDeviceCount)(&count);
    return err == gcxx::driver::deviceErrSuccess && count > 0;
  }

}  // namespace

class MemPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!gpuAvailable()) {
      GTEST_SKIP() << "No GPU device available; skipping pool ownership tests";
    }
  }
};

TEST_F(MemPoolTest, ConstructAndDestroy) {
  {
    const gcxx::MemPool pool(makeProps());
    EXPECT_NE(pool.getRawMemPool(), nullptr);
  }  // destroyed here; must not throw
}

TEST_F(MemPoolTest, MoveConstructorTransfersOwnership) {
  gcxx::MemPool pool1(makeProps());
  const auto raw = pool1.getRawMemPool();

  gcxx::MemPool pool2(std::move(pool1));

  EXPECT_EQ(pool1.getRawMemPool(), nullptr);
  EXPECT_EQ(pool2.getRawMemPool(), raw);
}

TEST_F(MemPoolTest, MoveAssignmentTransfersOwnership) {
  // pool1 is moved-from below; move-assignment needs a non-const source (copy
  // is deleted), so it must stay non-const despite the check's suggestion.
  gcxx::MemPool pool1(makeProps());  // NOLINT(misc-const-correctness)
  gcxx::MemPool pool2(makeProps());
  const auto raw1 = pool1.getRawMemPool();

  pool2 = std::move(pool1);

  EXPECT_EQ(pool1.getRawMemPool(), nullptr);
  EXPECT_EQ(pool2.getRawMemPool(), raw1);
}

TEST_F(MemPoolTest, ReleaseTransfersHandle) {
  gcxx::MemPool pool(makeProps());
  const auto raw = pool.getRawMemPool();

  const gcxx::MemPoolView view = pool.Release();

  EXPECT_EQ(pool.getRawMemPool(), nullptr);
  EXPECT_EQ(view.getRawMemPool(), raw);

  // Ownership transferred; destroy manually.
  gcxx::driver::deviceMemPoolDestroy(raw);
}

TEST_F(MemPoolTest, GetReturnsValidView) {
  const gcxx::MemPool pool(makeProps());
  const gcxx::MemPoolView view = pool.get();
  EXPECT_EQ(view.getRawMemPool(), pool.getRawMemPool());
}

TEST_F(MemPoolTest, SelfMoveKeepsOwnership) {
  gcxx::MemPool pool(makeProps());
  const auto raw = pool.getRawMemPool();

  pool = std::move(pool);  // NOLINT(clang-analyzer-cplusplus.Move)

  EXPECT_EQ(pool.getRawMemPool(), raw);
}

TEST_F(MemPoolTest, DestroyIsIdempotentOnReleased) {
  gcxx::MemPool pool(makeProps());
  const gcxx::MemPoolView view = pool.Release();
  pool.destroy();  // no-op on null handle after Release
  gcxx::driver::deviceMemPoolDestroy(view.getRawMemPool());
}
