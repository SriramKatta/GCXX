// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Coverage for heterogeneous_iterator: the buffer's iterator type, its
// random-access mechanics, dereference from host code (host_accessible buffer),
// and interop with std algorithms (std::fill / std::accumulate). Uses a host
// mock so the suite runs without a GPU.
//
// Accessibility-matrix coverage: the iterator's dereference is space-restricted
// (host-only / device-only / both) while every non-dereference operation is
// usable from both host and device. The device half of the matrix is exercised
// by deref_device_iter, a __global__ kernel whose body nvcc type-checks on the
// device path (so the test compiles even where no GPU is present) and which is
// launched GPU-guarded below for an end-to-end round-trip.
#include "tests_common.hpp"

#include <cstddef>
#include <cstdlib>
#include <numeric>
#include <type_traits>

#include <gcxx/api.hpp>

namespace {

  struct host_mock_resource {
    using properties = gcxx::TypeSet<gcxx::host_accessible>;
    void* allocate(gcxx::StreamView, std::size_t n) { return std::malloc(n); }
    void deallocate(gcxx::StreamView, void* p) { std::free(p); }
  };

  using host_buf = gcxx::buffer<int, gcxx::host_accessible>;

  void raw_fill(host_buf& b, int start) {
    for (std::size_t i = 0; i < b.size(); ++i)
      b.data()[i] = start + static_cast<int>(i);
  }

  // ─────────────── device-space access proof (compile + runtime)
  // ────────────── A __global__ entry point: nvcc compiles its body on the
  // device path, so the deref of a device_accessible iterator, the device-side
  // deref of a host_device (managed) iterator, and host-only-iterator
  // *arithmetic* (never its deref) are all type-checked — proving deref is
  // space-restricted while non-deref operations are always available. The
  // GPU-guarded test below launches it for an end-to-end round-trip; where no
  // GPU is present the body is still compiled (the restriction guarantee), the
  // launch is just skipped.
  __global__ void deref_device_iter(const int* in, int* out) {
    // device_accessible iterator: deref + subscript reachable from device code.
    // const CvTp so it binds the read-only device input (const int*).
    gcxx::heterogeneous_iterator<const int, gcxx::device_accessible> dit(in);
    const int a = *dit;    // device-only deref, on device: OK
    const int b = dit[2];  // device-only subscript, on device: OK

    // host_device (managed) iterator: deref reachable from device code over the
    // same device storage, plus arithmetic.
    gcxx::heterogeneous_iterator<const int, gcxx::host_accessible,
                                 gcxx::device_accessible>
      mit(in);
    const int c = *(mit + 1);  // managed deref + arithmetic, on device: OK

    *out = a + b + c;

    // host_accessible iterator: arithmetic is legal on the device (touches no
    // memory). Its deref is host-only and is deliberately NOT used — that is
    // the type-safe restriction; dereferencing `hit` here would not compile.
    gcxx::heterogeneous_iterator<int, gcxx::host_accessible> hit(nullptr);
    ++hit;
    (void)((hit + 1) - hit);
  }

}  // namespace

// =============================================================================
// buffer<VT, Properties...>::iterator is heterogeneous_iterator<VT,
// Properties...>.
// =============================================================================
TEST(HeterogeneousIteratorTest, BufferIteratorIsHeterogeneous) {
  static_assert(
    std::is_same_v<host_buf::iterator,
                   gcxx::heterogeneous_iterator<int, gcxx::host_accessible>>);
  static_assert(
    std::is_same_v<
      host_buf::const_iterator,
      gcxx::heterogeneous_iterator<const int, gcxx::host_accessible>>);
}

// =============================================================================
// Dereference + iteration via begin()/end() from host code.
// =============================================================================
TEST(HeterogeneousIteratorTest, IterateAndDereference) {
  host_buf b(gcxx::StreamView::Null(), host_mock_resource{}, std::size_t{4},
             gcxx::no_init);
  raw_fill(b, 10);

  int sum = 0;
  for (auto it = b.begin(); it != b.end(); ++it)
    sum += *it;
  EXPECT_EQ(sum, 10 + 11 + 12 + 13);
}

// =============================================================================
// Random-access mechanics: [], +, -, ++/--.
// =============================================================================
TEST(HeterogeneousIteratorTest, RandomAccessOps) {
  host_buf b(gcxx::StreamView::Null(), host_mock_resource{}, std::size_t{8},
             gcxx::no_init);
  raw_fill(b, 0);

  auto it = b.begin();
  EXPECT_EQ(*it, 0);
  EXPECT_EQ(it[3], 3);
  EXPECT_EQ(*(it + 5), 5);
  EXPECT_EQ((b.end() - b.begin()), 8);

  ++it;
  EXPECT_EQ(*it, 1);
  --it;
  EXPECT_EQ(*it, 0);
  it += 4;
  EXPECT_EQ(*it, 4);
  it -= 2;
  EXPECT_EQ(*it, 2);
}

// =============================================================================
// std algorithms work through the iterators (contiguous random-access).
// =============================================================================
TEST(HeterogeneousIteratorTest, StdAlgorithmsInterop) {
  host_buf b(gcxx::StreamView::Null(), host_mock_resource{}, std::size_t{8},
             gcxx::no_init);
  std::fill(b.begin(), b.end(), 7);
  EXPECT_EQ(*b.begin(), 7);
  EXPECT_EQ(*(b.end() - 1), 7);

  const int total = std::accumulate(b.begin(), b.end(), 0);
  EXPECT_EQ(total, 7 * 8);
}

// =============================================================================
// Member-type traits across the accessibility matrix. The CvTp qualifier maps
// to const-qualified pointer/reference; every instantiation is trivially
// copyable (one pointer of storage), so it is passable to a kernel by value.
// =============================================================================
TEST(HeterogeneousIteratorTest, MemberTypeTraitsAcrossAccessibility) {
  using d_iter = gcxx::heterogeneous_iterator<int, gcxx::device_accessible>;
  using d_citer =
    gcxx::heterogeneous_iterator<const int, gcxx::device_accessible>;
  using h_citer =
    gcxx::heterogeneous_iterator<const int, gcxx::host_accessible>;

  static_assert(std::is_same_v<d_iter::pointer, int*>);
  static_assert(std::is_same_v<d_iter::reference, int&>);
  static_assert(std::is_same_v<d_citer::pointer, const int*>);
  static_assert(std::is_same_v<d_citer::reference, const int&>);
  static_assert(std::is_same_v<h_citer::reference, const int&>);

  static_assert(std::is_trivially_copyable_v<d_iter>);
  static_assert(std::is_trivially_copyable_v<gcxx::heterogeneous_iterator<
                  int, gcxx::host_accessible, gcxx::device_accessible>>);
}

// =============================================================================
// Host-space dereference (host_accessible + host_device) and cross-space
// arithmetic. A device_accessible iterator's arithmetic is usable from host
// code because it touches no memory; its deref is device-only and is
// intentionally never used here — adding `*dit` would be a compile error.
// =============================================================================
TEST(HeterogeneousIteratorTest, HostSpaceDerefAndCrossSpaceArithmetic) {
  int raw[4] = {1, 2, 3, 4};

  gcxx::heterogeneous_iterator<int, gcxx::host_accessible> hit(raw);
  EXPECT_EQ(*hit, 1);
  EXPECT_EQ(hit[3], 4);
  auto h2 = hit + 2;
  EXPECT_EQ(*h2, 3);
  EXPECT_EQ(h2 - hit, 2);

  gcxx::heterogeneous_iterator<int, gcxx::host_accessible,
                               gcxx::device_accessible>
    mit(raw);
  EXPECT_EQ(*mit, 1);
  EXPECT_EQ(*(mit + 1), 2);

  gcxx::heterogeneous_iterator<int, gcxx::device_accessible> dit(raw);
  auto d2 = dit + 5;
  auto d3 = d2 - 2;
  EXPECT_EQ(d2 - dit, 5);
  EXPECT_NE(dit, d2);
  (void)d3;
}

// =============================================================================
// End-to-end device dereference: launch deref_device_iter over device storage
// and read the result back. Skipped on GPU-less hosts (the compile-time device
// guarantees above still hold). in = {10, 20, 30} → out = 10 + 30 + 20 = 60.
// =============================================================================
TEST(HeterogeneousIteratorTest, DeviceDerefFromDeviceKernel) {
  if (!gcxx::testing::haveCudaDevice())
    GTEST_SKIP() << "no CUDA device";

  auto in          = gcxx::make_device_unique_ptr<int>(3);
  auto out         = gcxx::make_device_unique_ptr<int>(1);
  int host_in[3]   = {10, 20, 30};
  int* host_in_ptr = host_in;  // Copy needs pointers (arrays don't decay here).
  gcxx::Copy(in.get(), host_in_ptr, std::size_t{3});

  deref_device_iter<<<1, 1>>>(in.get(), out.get());
  gcxx::driver::streamSynchronize(nullptr);

  int host_out = 0;
  gcxx::Copy(&host_out, out.get(), std::size_t{1});
  EXPECT_EQ(host_out, 60);
}
