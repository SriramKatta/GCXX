// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta

#include "tests_common.hpp"

#include <cstddef>

#include <gcxx/sparse_api.hpp>

namespace {
  // Minimal backend-agnostic device-buffer guard for descriptor wiring tests.
  // cuSPARSE/hipSPARSE only *store* these pointers at creation; they are not
  // dereferenced, so the contents are irrelevant — we just need valid device
  // addresses.
  struct DeviceBuf {
    void* p{nullptr};

    explicit DeviceBuf(std::size_t bytes) {
      (void)::GCXX_RUNTIME_BACKEND(Malloc)(&p, bytes);
    }

    ~DeviceBuf() {
      if (p != nullptr) {
        (void)::GCXX_RUNTIME_BACKEND(Free)(p);
      }
    }

    DeviceBuf(const DeviceBuf&)            = delete;
    DeviceBuf& operator=(const DeviceBuf&) = delete;

    auto get() const noexcept -> void* { return p; }
  };
}  // namespace

// Compile-time sanity: every owning descriptor is-a its non-owning view.
static_assert(std::is_base_of_v<gcxx::SpMatDescrView, gcxx::SpMatDescr>, "");
static_assert(std::is_base_of_v<gcxx::DnMatDescrView, gcxx::DnMatDescr>, "");
static_assert(std::is_base_of_v<gcxx::DnVecDescrView, gcxx::DnVecDescr>, "");
static_assert(std::is_base_of_v<gcxx::SpVecDescrView, gcxx::SpVecDescr>, "");

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║                          Sparse matrix (SpMat)                           ║
// ╚══════════════════════════════════════════════════════════════════════════╝

TEST(SpMatDescr, CreateCsrAndIntrospect) {
  if (!gcxx::testing::haveCudaDevice()) {
    GTEST_SKIP() << "No CUDA device available";
  }
  DeviceBuf rowOffsets(4 * sizeof(int));
  DeviceBuf colInd(3 * sizeof(int));
  DeviceBuf values(3 * sizeof(float));
  auto mat = gcxx::SpMatDescr::from_csr(
    3, 3, 3, rowOffsets.get(), colInd.get(), values.get(),
    gcxx::driver::deviceSparseIndex32I, gcxx::driver::deviceSparseIndex32I,
    gcxx::driver::deviceSparseIndexBaseZero, gcxx::driver::deviceValueTypeF32);
  EXPECT_NE(mat.getDescriptor(), nullptr);
  EXPECT_EQ(mat.getFormat(), gcxx::driver::deviceSparseFormatCsr);
  EXPECT_EQ(mat.getIndexBase(), gcxx::driver::deviceSparseIndexBaseZero);
}

TEST(SpMatDescr, CreateCooAndMove) {
  if (!gcxx::testing::haveCudaDevice()) {
    GTEST_SKIP() << "No CUDA device available";
  }
  DeviceBuf rowInd(3 * sizeof(int));
  DeviceBuf colInd(3 * sizeof(int));
  DeviceBuf values(3 * sizeof(float));
  auto mat = gcxx::SpMatDescr::from_coo(
    3, 3, 3, rowInd.get(), colInd.get(), values.get(),
    gcxx::driver::deviceSparseIndex32I, gcxx::driver::deviceSparseIndexBaseZero,
    gcxx::driver::deviceValueTypeF32);
  ASSERT_NE(mat.getDescriptor(), nullptr);
  EXPECT_EQ(mat.getFormat(), gcxx::driver::deviceSparseFormatCoo);

  auto raw = mat.getDescriptor();
  gcxx::SpMatDescr moved{std::move(mat)};
  EXPECT_EQ(moved.getDescriptor(), raw);
  EXPECT_EQ(mat.getDescriptor(), nullptr);  // NOLINT(bugprone-use-after-move)
}

TEST(SpMatDescr, ReleaseYieldsView) {
  if (!gcxx::testing::haveCudaDevice()) {
    GTEST_SKIP() << "No CUDA device available";
  }
  DeviceBuf rowOffsets(4 * sizeof(int));
  DeviceBuf colInd(3 * sizeof(int));
  DeviceBuf values(3 * sizeof(float));
  auto mat = gcxx::SpMatDescr::from_csr(
    3, 3, 3, rowOffsets.get(), colInd.get(), values.get(),
    gcxx::driver::deviceSparseIndex32I, gcxx::driver::deviceSparseIndex32I,
    gcxx::driver::deviceSparseIndexBaseZero, gcxx::driver::deviceValueTypeF32);
  auto raw  = mat.getDescriptor();
  auto view = mat.release();
  EXPECT_EQ(view.getDescriptor(), raw);
  EXPECT_EQ(mat.getDescriptor(), nullptr);
  [[maybe_unused]] auto adopted = gcxx::SpMatDescr::from_native_descriptor(raw);
}

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║                           Dense matrix (DnMat)                           ║
// ╚══════════════════════════════════════════════════════════════════════════╝

TEST(DnMatDescr, CreateAndDestroy) {
  if (!gcxx::testing::haveCudaDevice()) {
    GTEST_SKIP() << "No CUDA device available";
  }
  DeviceBuf values(6 * sizeof(float));
  auto mat = gcxx::DnMatDescr::from_dense(2, 3, 2, values.get(),
                                          gcxx::driver::deviceValueTypeF32,
                                          gcxx::driver::deviceSparseOrderCol);
  EXPECT_NE(mat.getDescriptor(), nullptr);
}

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║                          Sparse vector (SpVec)                           ║
// ╚══════════════════════════════════════════════════════════════════════════╝

TEST(SpVecDescr, CreateAndMove) {
  if (!gcxx::testing::haveCudaDevice()) {
    GTEST_SKIP() << "No CUDA device available";
  }
  DeviceBuf indices(3 * sizeof(int));
  DeviceBuf values(3 * sizeof(float));
  auto vec = gcxx::SpVecDescr::from_sparse(
    8, 3, indices.get(), values.get(), gcxx::driver::deviceSparseIndex32I,
    gcxx::driver::deviceSparseIndexBaseZero, gcxx::driver::deviceValueTypeF32);
  ASSERT_NE(vec.getDescriptor(), nullptr);

  auto raw = vec.getDescriptor();
  gcxx::SpVecDescr moved{std::move(vec)};
  EXPECT_EQ(moved.getDescriptor(), raw);
  EXPECT_EQ(vec.getDescriptor(), nullptr);  // NOLINT(bugprone-use-after-move)
}

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║                           Dense vector (DnVec)                           ║
// ╚══════════════════════════════════════════════════════════════════════════╝

TEST(DnVecDescr, CreateAndDestroy) {
  if (!gcxx::testing::haveCudaDevice()) {
    GTEST_SKIP() << "No CUDA device available";
  }
  DeviceBuf values(4 * sizeof(float));
  auto vec = gcxx::DnVecDescr::from_dense(4, values.get(),
                                          gcxx::driver::deviceValueTypeF32);
  EXPECT_NE(vec.getDescriptor(), nullptr);
}
