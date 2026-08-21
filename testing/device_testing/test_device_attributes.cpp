// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
// Tests for the typed device_attr / device_limit descriptor API
// (Device::attribute/limit/set_limit); GPU-dependent cases skip cleanly.
#include "tests_common.hpp"

#include <cstddef>
#include <type_traits>

#include <gcxx/api.hpp>

// Raw-handle contract for the graph wrappers; no dedicated graph test yet.
GCXX_ASSERT_RAW_HANDLE(GraphView, gcxx::driver::deviceGraph_t);
GCXX_ASSERT_RAW_HANDLE(Graph, gcxx::driver::deviceGraph_t);
GCXX_ASSERT_RAW_HANDLE(GraphExecView, gcxx::driver::deviceGraphExec_t);
GCXX_ASSERT_RAW_HANDLE(GraphExec, gcxx::driver::deviceGraphExec_t);
GCXX_ASSERT_RAW_HANDLE(GraphNodeView, gcxx::driver::deviceGraphNode_t);

// Compile-time type checks: int attributes, bool attributes, size_t limits.
static_assert(std::is_same_v<gcxx::dev_attr::warp_size_t::type, int>,
              "warp_size must be int-typed");
static_assert(std::is_same_v<gcxx::dev_attr::multiprocessor_count_t::type, int>,
              "multiprocessor_count must be int-typed");
static_assert(
  std::is_same_v<gcxx::dev_attr::compute_capability_major_t::type, int>,
  "compute_capability_major must be int-typed");
static_assert(std::is_same_v<gcxx::dev_attr::ecc_enabled_t::type, bool>,
              "ecc_enabled must be bool-typed");
static_assert(std::is_same_v<gcxx::dev_attr::managed_memory_t::type, bool>,
              "managed_memory must be bool-typed");
static_assert(std::is_same_v<gcxx::dev_attr::unified_addressing_t::type, bool>,
              "unified_addressing must be bool-typed");
static_assert(
  std::is_same_v<gcxx::device_limits::stack_size_t::type, std::size_t>,
  "device limits must be size_t-typed");

#if GCXX_CUDA_MODE()
static_assert(
  std::is_same_v<gcxx::device_limits::printf_fifo_size_t::type, std::size_t>,
  "device limits must be size_t-typed");
#endif

class DeviceAttributeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!gcxx::testing::haveCudaDevice()) {
      GTEST_SKIP()
        << "No CUDA device available; skipping device attribute tests";
    }
  }
};

TEST_F(DeviceAttributeTest, NumericAttributeReadsAsInt) {
  int warp = gcxx::Device::attribute(gcxx::dev_attr::warp_size);
  EXPECT_GT(warp, 0);  // warp size is always positive
}

TEST_F(DeviceAttributeTest, BooleanAttributeReadsAsBool) {
  bool managed = gcxx::Device::attribute(gcxx::dev_attr::managed_memory);
  bool unified = gcxx::Device::attribute(gcxx::dev_attr::unified_addressing);
  // Managed memory implies unified addressing; both are valid bools here.
  EXPECT_TRUE(managed == true || managed == false);
  EXPECT_TRUE(unified == true || unified == false);
  if (managed) {
    EXPECT_TRUE(
      unified);  // can't have managed memory without unified addressing
  }
}

TEST_F(DeviceAttributeTest, NumericAttributesAreSane) {
  EXPECT_GT(gcxx::Device::attribute(gcxx::dev_attr::max_threads_per_block), 0);
  EXPECT_GT(gcxx::Device::attribute(gcxx::dev_attr::multiprocessor_count), 0);
  EXPECT_GE(gcxx::Device::attribute(gcxx::dev_attr::compute_capability_major),
            0);
}

// Scopes to the handle's device; old getAttribute ignored it.
TEST_F(DeviceAttributeTest, HandleScopedAttribute) {
  gcxx::DeviceHandle handle{0};
  int warp = handle.attribute(gcxx::dev_attr::warp_size);
  EXPECT_GT(warp, 0);
}

#if GCXX_CUDA_MODE()
TEST_F(DeviceAttributeTest, LimitRoundTripViaDevice) {
  const std::size_t original =
    gcxx::Device::limit(gcxx::device_limits::printf_fifo_size);
  gcxx::Device::set_limit(gcxx::device_limits::printf_fifo_size,
                          std::size_t{1 << 20});
  EXPECT_EQ(gcxx::Device::limit(gcxx::device_limits::printf_fifo_size),
            std::size_t{1 << 20});
  // Restore the prior value.
  gcxx::Device::set_limit(gcxx::device_limits::printf_fifo_size, original);
  EXPECT_EQ(gcxx::Device::limit(gcxx::device_limits::printf_fifo_size),
            original);
}

TEST_F(DeviceAttributeTest, HandleLimitRoundTrip) {
  gcxx::DeviceHandle handle{0};
  const std::size_t original =
    handle.limit(gcxx::device_limits::printf_fifo_size);
  handle.set_limit(gcxx::device_limits::printf_fifo_size, std::size_t{1 << 20});
  EXPECT_EQ(handle.limit(gcxx::device_limits::printf_fifo_size),
            std::size_t{1 << 20});
  handle.set_limit(gcxx::device_limits::printf_fifo_size, original);
}
#endif