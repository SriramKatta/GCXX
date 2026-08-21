// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
// Coverage: heterogeneous_iterator mechanics, host-space deref, std
// algorithm interop; accessibility matrix incl. a GPU-guarded device kernel.
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

  // Compiled by nvcc even unlaunched; body proves the deref restrictions.
  __global__ void deref_device_iter(const int* in, int* out) {
    // Device-accessible iterator: deref + subscript usable from device code.
    gcxx::heterogeneous_iterator<const int, gcxx::device_accessible> dit(in);
    const int a = *dit;    // device-only deref, on device: OK
    const int b = dit[2];  // device-only subscript, on device: OK

    // Managed iterator: deref reachable from device code, plus arithmetic.
    gcxx::heterogeneous_iterator<const int, gcxx::host_accessible,
                                 gcxx::device_accessible>
      mit(in);
    const int c = *(mit + 1);  // managed deref + arithmetic, on device: OK

    *out = a + b + c;

    // Host-only iterator: on-device arithmetic is legal; its deref is not.
    gcxx::heterogeneous_iterator<int, gcxx::host_accessible> hit(nullptr);
    ++hit;
    (void)((hit + 1) - hit);
  }

}  // namespace

// Tags ARE the std tag types; typedefs declared directly (LWG 2438).
TEST(HeterogeneousIteratorTest, IteratorTraitsAliasAndTags) {
  static_assert(std::is_same_v<gcxx::random_access_iterator_tag,
                               std::random_access_iterator_tag>);
  static_assert(
    std::is_same_v<gcxx::input_iterator_tag, std::input_iterator_tag>);
  static_assert(
    std::is_same_v<gcxx::forward_iterator_tag, std::forward_iterator_tag>);
  static_assert(std::is_same_v<gcxx::bidirectional_iterator_tag,
                               std::bidirectional_iterator_tag>);
#if defined(__cpp_lib_contiguous_iterator) && \
  __cpp_lib_contiguous_iterator >= 201902L
  static_assert(std::is_same_v<gcxx::contiguous_iterator_tag,
                               std::contiguous_iterator_tag>);
#endif

  // gcxx::iterator_traits IS std::iterator_traits (pointers included).
  static_assert(std::is_same_v<gcxx::iterator_traits<int*>::value_type, int>);
  static_assert(std::is_same_v<gcxx::iterator_traits<int*>::difference_type,
                               std::ptrdiff_t>);
  static_assert(std::is_same_v<gcxx::iterator_traits<int*>::pointer, int*>);
  static_assert(std::is_same_v<gcxx::iterator_traits<int*>::reference, int&>);

  // All gcxx iterators report the gcxx (== std) tag directly.
  static_assert(std::is_same_v<host_buf::iterator::iterator_category,
                               gcxx::random_access_iterator_tag>);
  static_assert(std::is_same_v<gcxx::stride_iterator<int*>::iterator_category,
                               gcxx::random_access_iterator_tag>);
  static_assert(std::is_same_v<
                gcxx::reverse_iterator<host_buf::iterator>::iterator_category,
                gcxx::random_access_iterator_tag>);

  // The five typedefs are declared directly (no base class needed).
  using hit = gcxx::heterogeneous_iterator<int, gcxx::host_accessible>;
  static_assert(std::is_same_v<hit::value_type, int>);
  static_assert(std::is_same_v<hit::pointer, int*>);
  static_assert(std::is_same_v<hit::reference, int&>);
  static_assert(std::is_same_v<gcxx::iterator_traits<hit>::iterator_category,
                               gcxx::random_access_iterator_tag>);
  using const_hit =
    gcxx::heterogeneous_iterator<const int, gcxx::device_accessible>;
  static_assert(std::is_same_v<const_hit::value_type, int>);
  static_assert(std::is_same_v<const_hit::pointer, const int*>);
  static_assert(std::is_same_v<const_hit::reference, const int&>);
  static_assert(std::is_same_v<gcxx::stride_iterator<int*>::value_type, int>);
  static_assert(std::is_same_v<gcxx::stride_iterator<int*>::reference, int&>);
  static_assert(std::is_same_v<gcxx::reverse_iterator<hit>::value_type, int>);
  static_assert(std::is_same_v<gcxx::reverse_iterator<hit>::reference, int&>);
}

TEST(HeterogeneousIteratorTest, BufferIteratorAliasesStorage) {
  static_assert(
    std::is_same_v<host_buf::iterator,
                   gcxx::heterogeneous_iterator<int, gcxx::host_accessible>>);
  static_assert(
    std::is_same_v<host_buf::iterator,
                   gcxx::uninit_buffer<int, gcxx::host_accessible>::iterator>);
  static_assert(
    std::is_same_v<
      host_buf::const_iterator,
      gcxx::uninit_buffer<int, gcxx::host_accessible>::const_iterator>);
}

TEST(HeterogeneousIteratorTest, ReverseIteratorAliasesAndSemantics) {
  static_assert(
    std::is_same_v<host_buf::reverse_iterator,
                   gcxx::reverse_iterator<gcxx::heterogeneous_iterator<
                     int, gcxx::host_accessible>>>);
  static_assert(
    std::is_same_v<
      host_buf::reverse_iterator,
      gcxx::uninit_buffer<int, gcxx::host_accessible>::reverse_iterator>);
  static_assert(
    std::is_same_v<host_buf::const_reverse_iterator,
                   gcxx::reverse_iterator<gcxx::heterogeneous_iterator<
                     const int, gcxx::host_accessible>>>);

  host_buf b(gcxx::StreamView::Null(), host_mock_resource{}, std::size_t{5},
             gcxx::no_init);
  raw_fill(b, 0);  // elements 0 1 2 3 4

  // rbegin references the last element; indexing counts backwards.
  EXPECT_EQ(*b.rbegin(), 4);
  EXPECT_EQ(b.rbegin()[0], 4);
  EXPECT_EQ(b.rbegin()[2], 2);

  // [rbegin, rend) visits elements in reverse order.
  int expected = 4;
  for (auto it = b.rbegin(); it != b.rend(); ++it, --expected) {
    EXPECT_EQ(*it, expected);
  }
  EXPECT_EQ(std::accumulate(b.rbegin(), b.rend(), 0),
            std::accumulate(b.begin(), b.end(), 0));

  // Mechanics: ++ moves backwards, -- forwards, +/-/comparisons flipped.
  auto it = b.rbegin();
  it += 2;
  EXPECT_EQ(*it, 2);
  it -= 1;
  EXPECT_EQ(*it, 3);
  EXPECT_EQ(*(it + 1), 2);
  EXPECT_EQ(*(it - 1), 4);
  EXPECT_EQ(b.rend() - b.rbegin(), 5);
  EXPECT_TRUE(b.rbegin() < b.rend());
  --it;
  EXPECT_EQ(*it, 4);

  // Genericity: the adapter wraps ANY iterator — raw pointers included.
  int arr[5]{0, 1, 2, 3, 4};
  gcxx::reverse_iterator<int*> rp(arr + 5);
  EXPECT_EQ(*rp, 4);
  EXPECT_EQ(rp[3], 1);
  EXPECT_EQ(gcxx::reverse_iterator<int*>(arr) - rp, 5);  // rend - rbegin
}

TEST(HeterogeneousIteratorTest, IterateAndDereference) {
  host_buf b(gcxx::StreamView::Null(), host_mock_resource{}, std::size_t{4},
             gcxx::no_init);
  raw_fill(b, 10);

  int sum = 0;
  for (auto it = b.begin(); it != b.end(); ++it)
    sum += *it;
  EXPECT_EQ(sum, 10 + 11 + 12 + 13);
}

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

TEST(HeterogeneousIteratorTest, StdAlgorithmsInterop) {
  host_buf b(gcxx::StreamView::Null(), host_mock_resource{}, std::size_t{8},
             gcxx::no_init);
  std::fill(b.begin(), b.end(), 7);
  EXPECT_EQ(*b.begin(), 7);
  EXPECT_EQ(*(b.end() - 1), 7);

  const int total = std::accumulate(b.begin(), b.end(), 0);
  EXPECT_EQ(total, 7 * 8);
}

// One pointer of storage; trivially copyable, so passable to kernels by value.
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

// Device-it arithmetic is host-legal (touches no memory); *dit won't compile.
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

// in = {10, 20, 30} -> out = 10 + 30 + 20 = 60.
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
