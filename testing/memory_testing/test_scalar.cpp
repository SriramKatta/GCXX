// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
// Owning scalar container: property TypeSets, detection traits, SFINAE
// contracts, and value round-trips. Host-only paths run without a GPU;
// device/pinned/managed round-trips are GPU-gated.

#include "tests_common.hpp"

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include <gcxx/api.hpp>

namespace {

  // host_mock_resource (plain pageable host memory) comes from
  // tests_common.hpp.

  using host_scalar = gcxx::host_scalar<double>;
  using dev_scalar  = gcxx::device_scalar<double>;
  using pin_scalar  = gcxx::pinned_scalar<double>;

  using host_vec =
    gcxx::mdspan<double, gcxx::dextents<int, 1>, gcxx::layout_left,
                 gcxx::default_accessor<double>>;

  // Direct void_t detectors (not GCXX_DEFINE_IS_CALLABLE): the probed type
  // must be a template parameter of the detector, otherwise EDG evaluates
  // the decltype eagerly — a hard error on the deleted overloads instead of
  // SFINAE. Ctor contracts use std::is_constructible for the same reason.

  template <class T, class = void>
  struct set_value_async_lvalue_ok : std::false_type {};
  template <class T>
  struct set_value_async_lvalue_ok<
    T, std::void_t<decltype(std::declval<T&>().set_value_async(
         std::declval<const double&>()))>> : std::true_type {};

  template <class T, class = void>
  struct set_value_async_rvalue_ok : std::false_type {};
  template <class T>
  struct set_value_async_rvalue_ok<
    T, std::void_t<decltype(std::declval<T&>().set_value_async(3.0))>>
      : std::true_type {};

  template <class T, class = void>
  struct scaled_lvalue_ok : std::false_type {};
  template <class T>
  struct scaled_lvalue_ok<
    T, std::void_t<decltype(gcxx::scaled(std::declval<const T&>(),
                                         std::declval<const host_vec&>()))>>
      : std::true_type {};

  template <class T, class = void>
  struct scaled_temporary_ok : std::false_type {};
  template <class T>
  struct scaled_temporary_ok<
    T, std::void_t<decltype(gcxx::scaled(std::declval<T&&>(),
                                         std::declval<const host_vec&>()))>>
      : std::true_type {};

}  // namespace

TEST(ScalarTypeTest, AliasesCarryExpectedTypeSets) {
  static_assert(
    std::is_same_v<
      dev_scalar::properties,
      gcxx::TypeSet<gcxx::device_accessible, gcxx::device_allocated>>);
  static_assert(
    std::is_same_v<pin_scalar::properties,
                   gcxx::TypeSet<gcxx::host_accessible, gcxx::device_accessible,
                                 gcxx::pinned_allocated>>);
  // Kind tags make pinned and managed DISTINCT types: managed memory is
  // unified_allocated, not host_allocated, so it cannot name host-mode
  // positions statically.
  static_assert(!std::is_same_v<gcxx::managed_scalar<double>, pin_scalar>);
  static_assert(
    std::is_same_v<gcxx::host_scalar<double>::properties,
                   gcxx::TypeSet<gcxx::host_accessible, gcxx::host_allocated>>);
  static_assert(std::is_same_v<dev_scalar::value_type, double>);
}

TEST(ScalarTypeTest, OwningScalarDetection) {
  static_assert(gcxx::details_::is_owning_scalar_v<dev_scalar&>);
  static_assert(gcxx::details_::is_owning_scalar_v<const pin_scalar>);
  static_assert(gcxx::details_::is_owning_scalar_v<dev_scalar&&>);
  static_assert(!gcxx::details_::is_owning_scalar_v<double>);
  static_assert(
    !gcxx::details_::is_owning_scalar_v<gcxx::device_buffer<double>>);
  static_assert(
    !gcxx::details_::is_owning_scalar_v<gcxx::uninit_device_buffer<double>>);
  static_assert(
    !gcxx::details_::is_owning_scalar_v<gcxx::device_scalar_view<double>>);
}

TEST(ScalarSfinaeTest, MoveOnlyAndLifetimeContracts) {
  // Move-only; deep copies require an explicit stream.
  static_assert(!std::is_copy_constructible_v<host_scalar>);
  static_assert(!std::is_constructible_v<host_scalar, const host_scalar&>);
  static_assert(std::is_move_constructible_v<host_scalar>);
  // No stream-only ctor: an allocation always needs a resource.
  static_assert(!std::is_constructible_v<host_scalar, gcxx::StreamView>);

  // set_value_async rejects rvalue temporaries (async copy source would
  // dangle), accepts lvalues.
  static_assert(!set_value_async_rvalue_ok<host_scalar>::value);
  static_assert(set_value_async_lvalue_ok<host_scalar>::value);

  // scaled() accepts an owning scalar by lvalue (decaying to a non-owning
  // view factor); a temporary owner is deleted outright (the copied mdspan
  // would outlive it). Plain factors keep the generic path.
  static_assert(scaled_lvalue_ok<dev_scalar>::value);
  static_assert(!scaled_temporary_ok<dev_scalar>::value);
  static_assert(
    gcxx::is_scaled_accessor_v<typename std::remove_reference_t<
      decltype(gcxx::scaled(std::declval<const double&>(),
                            std::declval<const host_vec&>()))>::accessor_type>);
}

TEST(ScalarHostTest, HostOnlyRoundTrip) {
  host_scalar s(gcxx::StreamView::Null(), host_mock_resource{}, 7.0);

  EXPECT_EQ(s.size(), std::size_t{1});
  EXPECT_FALSE(s.empty());
  EXPECT_EQ(s.size_bytes(), sizeof(double));
  EXPECT_EQ(s.value(), 7.0);

  // host-accessible storage: direct write, no stream traffic.
  s.set_value(9.0);
  EXPECT_EQ(s.value(), 9.0);
  s.set_value_to_zero_async();
  EXPECT_EQ(s.value(), 0.0);

  // Deep copy on an explicit stream, borrowing the source's resource.
  s.set_value(4.0);
  host_scalar copy(s, gcxx::StreamView::Null());
  EXPECT_EQ(copy.value(), 4.0);

  // Move-only: the moved-from object remains valid (empty allocation).
  host_scalar moved(std::move(copy));
  EXPECT_EQ(moved.value(), 4.0);
}

TEST(ScalarHostTest, ScaledHostOnlyFactorResolvesToValue) {
  // A host-only scalar resolves to a plain host value factor at the scaled()
  // boundary (its storage is directly readable).
  host_scalar s(gcxx::StreamView::Null(), host_mock_resource{}, 3.0);

  std::vector<double> data{1.0, 2.0, 3.0};
  host_vec hv(data.data(), 3);

  auto view = gcxx::scaled(s, hv);
  using expected_accessor =
    gcxx::scaled_accessor<double, gcxx::default_accessor<double>>;
  static_assert(
    std::is_same_v<std::remove_reference_t<decltype(view)>::accessor_type,
                   expected_accessor>);
  EXPECT_EQ(view.accessor().scaling_factor(), 3.0);
}

TEST(ScalarDeviceTest, DeviceScalarRoundTrip) {
  GCXX_SKIP_WITHOUT_DEVICE();

  gcxx::Stream str;
  gcxx::DeviceHandle dev{0};

  auto alpha = gcxx::make_device_scalar<double>(dev, str, 2.0);
  EXPECT_EQ(alpha.stream().getRawHandle(), str.getRawHandle());
  // value() is the synchronizing accessor (D2H + stream sync).
  EXPECT_EQ(alpha.value(), 2.0);

  // set_value_async takes its source by const& (rvalues are deleted: the
  // in-flight async copy would outlive a temporary).
  const double five = 5.0;
  alpha.set_value_async(five, str);
  str.sync();
  EXPECT_EQ(alpha.value(), 5.0);

  alpha.set_value(6.0);
  EXPECT_EQ(alpha.value(), 6.0);

  alpha.set_value_to_zero_async(str);
  str.sync();
  EXPECT_EQ(alpha.value(), 0.0);

  // Deep copy on an explicit stream from the default device pool.
  alpha.set_value(8.0);
  decltype(alpha) beta(alpha, str, gcxx::device_default_memory_pool(dev));
  EXPECT_EQ(beta.value(), 8.0);

  auto moved = std::move(beta);
  EXPECT_EQ(moved.value(), 8.0);
}

TEST(ScalarDeviceTest, PinnedScalarRoundTrip) {
  GCXX_SKIP_WITHOUT_DEVICE();

  gcxx::Stream str;
  auto alpha = gcxx::make_pinned_scalar<double>(str, 3.0);

  // Pinned storage is host+device accessible: direct host read/write.
  EXPECT_EQ(alpha.value(), 3.0);
  alpha.set_value(4.0);
  EXPECT_EQ(alpha.value(), 4.0);

  // Still a valid device pointer-mode factor (device_accessible): the
  // accessor stores the copyable device_scalar_view, never the owner.
  std::vector<double> data{1.0, 2.0, 3.0};
  host_vec hv(data.data(), 3);
  auto view = gcxx::scaled(alpha, hv);
  using expected_accessor =
    gcxx::scaled_accessor<gcxx::device_scalar_view<double>,
                          gcxx::default_accessor<double>>;
  static_assert(
    std::is_same_v<std::remove_reference_t<decltype(view)>::accessor_type,
                   expected_accessor>);
  EXPECT_EQ(view.accessor().scaling_factor().ptr, alpha.data());
}

// Managed pools exist from CUDA 13.0 (managed_default_memory_pool gate in
// mempool/default_memory_pools.hpp); on older toolchains the managed kind
// shares the pinned test's instantiation (host+device accessible) and only
// the resource differs.
#if GCXX_HAS_MANAGED_POOLS
TEST(ScalarDeviceTest, ManagedScalarRoundTrip) {
  GCXX_SKIP_WITHOUT_DEVICE();

  gcxx::Stream str;
  gcxx::ManagedMemPool pool{};
  // ManagedMemPool itself is move-only; scalars keep a copyable view.
  gcxx::managed_scalar<double> alpha(str, pool.as_ref(), 6.0);

  EXPECT_EQ(alpha.value(), 6.0);
  alpha.set_value(7.0);
  EXPECT_EQ(alpha.value(), 7.0);
}
#endif  // GCXX_HAS_MANAGED_POOLS
