// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta

#include "tests_common.hpp"

#include <cstddef>

#include <gcxx/sparse_api.hpp>

// raw_handle_type contract (see tests_common.hpp).
GCXX_ASSERT_RAW_HANDLE(DnVecDescrView, gcxx::driver::deviceDnVecDescr_t);
GCXX_ASSERT_RAW_HANDLE(DnVecDescr, gcxx::driver::deviceDnVecDescr_t);
GCXX_ASSERT_RAW_HANDLE(SpVecDescrView, gcxx::driver::deviceSpVecDescr_t);
GCXX_ASSERT_RAW_HANDLE(SpVecDescr, gcxx::driver::deviceSpVecDescr_t);
GCXX_ASSERT_RAW_HANDLE(DnMatDescrView, gcxx::driver::deviceDnMatDescr_t);
GCXX_ASSERT_RAW_HANDLE(DnMatDescr, gcxx::driver::deviceDnMatDescr_t);
GCXX_ASSERT_RAW_HANDLE(SpMatDescrView, gcxx::driver::deviceSpMatDescr_t);
GCXX_ASSERT_RAW_HANDLE(SpMatDescr, gcxx::driver::deviceSpMatDescr_t);

namespace {
  // Backend only stores these pointers at creation (never derefs them).
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

TEST(SpMatDescr, CreateCsrAndIntrospect) {
  GCXX_SKIP_WITHOUT_DEVICE();
  DeviceBuf rowOffsets(4 * sizeof(int));
  DeviceBuf colInd(3 * sizeof(int));
  DeviceBuf values(3 * sizeof(float));
  auto mat = gcxx::SpMatDescr::from_csr(
    3, 3, 3, rowOffsets.get(), colInd.get(), values.get(),
    gcxx::driver::deviceSparseIndex32I, gcxx::driver::deviceSparseIndex32I,
    gcxx::driver::deviceSparseIndexBaseZero, gcxx::driver::deviceValueTypeF32);
  EXPECT_NE(mat.getRawHandle(), nullptr);
  EXPECT_EQ(mat.getFormat(), gcxx::driver::deviceSparseFormatCsr);
  EXPECT_EQ(mat.getIndexBase(), gcxx::driver::deviceSparseIndexBaseZero);
}

TEST(SpMatDescr, CreateCooAndMove) {
  GCXX_SKIP_WITHOUT_DEVICE();
  DeviceBuf rowInd(3 * sizeof(int));
  DeviceBuf colInd(3 * sizeof(int));
  DeviceBuf values(3 * sizeof(float));
  auto mat = gcxx::SpMatDescr::from_coo(
    3, 3, 3, rowInd.get(), colInd.get(), values.get(),
    gcxx::driver::deviceSparseIndex32I, gcxx::driver::deviceSparseIndexBaseZero,
    gcxx::driver::deviceValueTypeF32);
  ASSERT_NE(mat.getRawHandle(), nullptr);
  EXPECT_EQ(mat.getFormat(), gcxx::driver::deviceSparseFormatCoo);

  auto raw = mat.getRawHandle();
  gcxx::SpMatDescr moved{std::move(mat)};
  EXPECT_EQ(moved.getRawHandle(), raw);
  EXPECT_EQ(mat.getRawHandle(), nullptr);  // NOLINT(bugprone-use-after-move)
}

TEST(SpMatDescr, ReleaseYieldsView) {
  GCXX_SKIP_WITHOUT_DEVICE();
  DeviceBuf rowOffsets(4 * sizeof(int));
  DeviceBuf colInd(3 * sizeof(int));
  DeviceBuf values(3 * sizeof(float));
  auto mat = gcxx::SpMatDescr::from_csr(
    3, 3, 3, rowOffsets.get(), colInd.get(), values.get(),
    gcxx::driver::deviceSparseIndex32I, gcxx::driver::deviceSparseIndex32I,
    gcxx::driver::deviceSparseIndexBaseZero, gcxx::driver::deviceValueTypeF32);
  auto raw  = mat.getRawHandle();
  auto view = mat.release();
  EXPECT_EQ(view.getRawHandle(), raw);
  EXPECT_EQ(mat.getRawHandle(), nullptr);
  [[maybe_unused]] auto adopted = gcxx::SpMatDescr::from_native_descriptor(raw);
}

TEST(DnMatDescr, CreateAndDestroy) {
  GCXX_SKIP_WITHOUT_DEVICE();
  DeviceBuf values(6 * sizeof(float));
  auto mat = gcxx::DnMatDescr::from_dense(2, 3, 2, values.get(),
                                          gcxx::driver::deviceValueTypeF32,
                                          gcxx::driver::deviceSparseOrderCol);
  EXPECT_NE(mat.getRawHandle(), nullptr);
}

TEST(SpVecDescr, CreateAndMove) {
  GCXX_SKIP_WITHOUT_DEVICE();
  DeviceBuf indices(3 * sizeof(int));
  DeviceBuf values(3 * sizeof(float));
  auto vec = gcxx::SpVecDescr::from_sparse(
    8, 3, indices.get(), values.get(), gcxx::driver::deviceSparseIndex32I,
    gcxx::driver::deviceSparseIndexBaseZero, gcxx::driver::deviceValueTypeF32);
  ASSERT_NE(vec.getRawHandle(), nullptr);

  auto raw = vec.getRawHandle();
  gcxx::SpVecDescr moved{std::move(vec)};
  EXPECT_EQ(moved.getRawHandle(), raw);
  EXPECT_EQ(vec.getRawHandle(), nullptr);  // NOLINT(bugprone-use-after-move)
}

TEST(DnVecDescr, CreateAndDestroy) {
  GCXX_SKIP_WITHOUT_DEVICE();
  DeviceBuf values(4 * sizeof(float));
  auto vec = gcxx::DnVecDescr::from_dense(4, values.get(),
                                          gcxx::driver::deviceValueTypeF32);
  EXPECT_NE(vec.getRawHandle(), nullptr);
}
