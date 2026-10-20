// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
// Host-only tests for the BLAS view helpers (transposed/scaled) and
// stride-based op/leading-dim/output inference. Pure metadata; no GPU.

#include "tests_common.hpp"

#include <array>
#include <type_traits>
#include <vector>

#include <gcxx/blas_api.hpp>
#include <gcxx/runtime/memory/spans/mdspan/mdspan.hpp>

namespace {

  using dextents2d = gcxx::dextents<int, 2>;
  using def_acc_d  = gcxx::default_accessor<double>;
  using dev_acc_d  = gcxx::device_accessor<def_acc_d>;

  template <class Layout>
  using mat2d = gcxx::mdspan<double, dextents2d, Layout, def_acc_d>;

  // Device-view counterpart required by the BLAS inference helpers.
  template <class Layout>
  using dmat2d = gcxx::mdspan<double, dextents2d, Layout, dev_acc_d>;

  template <class V>
  auto infer(const V& v) {
    return gcxx::blas::details_::infer_blas_matrix_view(v);
  }

}  // namespace

TEST(BlasTransposed, SwapsExtentsAndStrides_LayoutLeft) {
  std::vector<double> buf(3 * 4);
  mat2d<gcxx::layout_left> a(buf.data(), 3, 4);  // strides (1, 3)

  auto at = gcxx::transposed(a);
  EXPECT_EQ(at.extent(0), 4);
  EXPECT_EQ(at.extent(1), 3);
  EXPECT_EQ(at.mapping().stride(0), 3);
  EXPECT_EQ(at.mapping().stride(1), 1);
}

TEST(BlasTransposed, SwapsExtentsAndStrides_LayoutRight) {
  std::vector<double> buf(3 * 4);
  mat2d<gcxx::layout_right> a(buf.data(), 3, 4);  // strides (4, 1)

  auto at = gcxx::transposed(a);
  EXPECT_EQ(at.extent(0), 4);
  EXPECT_EQ(at.extent(1), 3);
  EXPECT_EQ(at.mapping().stride(0), 1);
  EXPECT_EQ(at.mapping().stride(1), 4);
}

TEST(BlasTransposed, RoundTrips_LayoutLeft) {
  std::vector<double> buf(3 * 4);
  mat2d<gcxx::layout_left> a(buf.data(), 3, 4);

  auto att = gcxx::transposed(gcxx::transposed(a));
  EXPECT_EQ(att.extent(0), a.extent(0));
  EXPECT_EQ(att.extent(1), a.extent(1));
  EXPECT_EQ(att.mapping().stride(0), a.mapping().stride(0));
  EXPECT_EQ(att.mapping().stride(1), a.mapping().stride(1));
}

// Type-level guarantee the dextents-only version could not provide.
TEST(BlasTransposed, PreservesStaticExtentsAndType) {
  double buf[12] = {};
  gcxx::mdspan<double, gcxx::extents<int, 3, 4>, gcxx::layout_left,
               gcxx::default_accessor<double>>
    a(buf);

  auto at = gcxx::transposed(a);
  static_assert(
    std::is_same_v<decltype(at)::extents_type, gcxx::extents<int, 4, 3>>,
    "static extents transpose 3x4 -> 4x3");
  static_assert(std::is_same_v<decltype(at)::layout_type, gcxx::layout_right>,
                "layout_left transposes to layout_right");
  EXPECT_EQ(at.extent(0), 4);
  EXPECT_EQ(at.extent(1), 3);

  auto att = gcxx::transposed(at);
  static_assert(
    std::is_same_v<decltype(att)::extents_type, gcxx::extents<int, 3, 4>>,
    "round-trip restores original static extents");
  static_assert(std::is_same_v<decltype(att)::layout_type, gcxx::layout_left>,
                "round-trip restores original layout");
}

TEST(BlasScaled, ElementAccessIsScaled) {
  std::vector<double> buf{1.0, -2.0, 3.0, -4.0};
  gcxx::mdspan<double, gcxx::dextents<int, 1>> x(buf.data(), 4);

  auto xs = gcxx::scaled(2.5, x);
  static_assert(std::is_same_v<decltype(xs)::element_type, double>,
                "scaled views keep the inner element type (BLAS dispatch "
                "keys on it)");
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(xs[i], 2.5 * buf[static_cast<std::size_t>(i)]);
  }
}

TEST(BlasScaled, PreservesMappingAndDataHandle) {
  std::vector<double> buf(3 * 4);
  mat2d<gcxx::layout_left> a(buf.data(), 3, 4);

  auto as = gcxx::scaled(2.0, a);
  EXPECT_EQ(as.data_handle(), a.data_handle());
  EXPECT_EQ(as.extent(0), 3);
  EXPECT_EQ(as.extent(1), 4);
  EXPECT_EQ(as.mapping().stride(0), 1);
  EXPECT_EQ(as.mapping().stride(1), 3);
}

TEST(BlasScaled, DeviceViewPropagatesThroughAccessor) {
  static_assert(
    gcxx::is_device_view_v<
      gcxx::scaled_accessor<double, gcxx::device_accessor<def_acc_d>>>,
    "a scaled device view is still a device view (BLAS operand gate)");
  static_assert(
    !gcxx::is_device_view_v<gcxx::scaled_accessor<double, def_acc_d>>,
    "a scaled host view is not a device view");
  static_assert(
    gcxx::is_host_view_v<
      gcxx::scaled_accessor<double, gcxx::host_accessor<def_acc_d>>>,
    "a scaled host_accessor view is a host view");
  static_assert(
    !gcxx::is_host_view_v<gcxx::scaled_accessor<double, def_acc_d>>,
    "a plain default_accessor view carries no space classification (and "
    "neither does its scaled wrapper)");
  static_assert(
    gcxx::is_scaled_accessor_v<gcxx::scaled_accessor<double, dev_acc_d>>,
    "identity trait");

  // and end-to-end through the view function on a device mdspan
  std::vector<double> buf(3 * 4);
  dmat2d<gcxx::layout_left> a(buf.data(), 3, 4);
  auto as = gcxx::scaled(1.5, a);
  static_assert(gcxx::is_device_view_v<decltype(as)::accessor_type>,
                "scaled(device view) still passes the BLAS operand gate");
  SUCCEED();
}

TEST(BlasScaled, StripScaledRecoversBaseView) {
  std::vector<double> buf{1.0, 2.0};
  gcxx::mdspan<double, gcxx::dextents<int, 1>> x(buf.data(), 2);

  auto xs  = gcxx::scaled(3.0, x);
  auto xs2 = gcxx::strip_scaled(xs);
  static_assert(!gcxx::is_scaled_accessor_v<decltype(xs2)::accessor_type>,
                "strip_scaled removes the scaled layer");
  EXPECT_EQ(xs2.data_handle(), x.data_handle());
  EXPECT_EQ(xs2[0], 1.0);

  auto idem = gcxx::strip_scaled(x);
  static_assert(!gcxx::is_scaled_accessor_v<decltype(idem)::accessor_type>,
                "strip_scaled is the identity on plain views");
  EXPECT_EQ(idem.data_handle(), x.data_handle());
}

TEST(BlasScaled, AlphaResolutionCombinesHostFactors) {
  using gcxx::scaled_accessor;
  using gcxx::blas::details_::alpha_resolution;
  using gcxx::blas::details_::combine_scaled_alpha;
  using gcxx::blas::details_::resolve_scaled_alpha;

  constexpr alpha_resolution<double> none{};
  static_assert(!none.from_device(), "default resolution is host");
  EXPECT_EQ(none.host_value, 1.0);

  scaled_accessor<double, def_acc_d> acc{2.0, def_acc_d{}};
  const auto r2 = resolve_scaled_alpha<double>(acc);
  EXPECT_EQ(r2.host_value, 2.0);
  EXPECT_FALSE(r2.from_device());

  const auto both =
    combine_scaled_alpha(resolve_scaled_alpha<double>(acc),
                         resolve_scaled_alpha<double>(acc), "test");
  EXPECT_EQ(both.host_value, 4.0);
  EXPECT_FALSE(both.from_device());
}

// Op inference keys on which axis is unit-stride; dims are the extents.
TEST(BlasOpInference, ColumnContiguous_IsOpN) {
  std::vector<double> buf(3 * 4);
  dmat2d<gcxx::layout_left> a(buf.data(), 3, 4);  // strides (1, 3)

  const auto info = infer(a);
  EXPECT_EQ(info.op, gcxx::driver::deviceBlasOpN);
  EXPECT_EQ(info.rows, 3);
  EXPECT_EQ(info.cols, 4);
  EXPECT_EQ(info.leading_dimension, 3);
}

TEST(BlasOpInference, RowContiguous_IsOpT) {
  std::vector<double> buf(3 * 4);
  dmat2d<gcxx::layout_right> a(buf.data(), 3, 4);  // strides (4, 1)

  const auto info = infer(a);
  EXPECT_EQ(info.op, gcxx::driver::deviceBlasOpT);
  EXPECT_EQ(info.rows, 3);
  EXPECT_EQ(info.cols, 4);
  EXPECT_EQ(info.leading_dimension, 4);
}

TEST(BlasOpInference, TransposedLayoutLeft_IsOpT) {
  std::vector<double> buf(3 * 4);
  dmat2d<gcxx::layout_left> a(buf.data(), 3, 4);

  const auto info = infer(gcxx::transposed(a));  // extents (4,3), strides (3,1)
  EXPECT_EQ(info.op, gcxx::driver::deviceBlasOpT);
  EXPECT_EQ(info.rows, 4);
  EXPECT_EQ(info.cols, 3);
  EXPECT_EQ(info.leading_dimension, 3);
}

TEST(BlasOpInference, TransposedLayoutRight_IsOpN) {
  std::vector<double> buf(3 * 4);
  dmat2d<gcxx::layout_right> a(buf.data(), 3, 4);

  const auto info = infer(gcxx::transposed(a));  // extents (4,3), strides (1,4)
  EXPECT_EQ(info.op, gcxx::driver::deviceBlasOpN);
  EXPECT_EQ(info.rows, 4);
  EXPECT_EQ(info.cols, 3);
  EXPECT_EQ(info.leading_dimension, 4);
}

// Strides (1,1) tie-break must yield op=N, not reject.
TEST(BlasOpInference, DegenerateUnitStrides_TieBreaksToOpN) {
  std::vector<double> buf(4);
  dmat2d<gcxx::layout_left> a(buf.data(), 1, 4);  // strides (1, 1)

  const auto info = infer(a);
  EXPECT_EQ(info.op, gcxx::driver::deviceBlasOpN);
  EXPECT_EQ(info.rows, 1);
  EXPECT_EQ(info.cols, 4);
  EXPECT_EQ(info.leading_dimension, 1);
}

// Strided subview rejected: throws in exception builds, aborts otherwise.
TEST(BlasOpInference, NeitherAxisUnit_Rejected) {
  std::vector<double> buf(100);
  dextents2d ext{4, 5};
  gcxx::layout_stride::mapping<dextents2d> map{ext, std::array<int, 2>{2, 7}};
  gcxx::mdspan<double, dextents2d, gcxx::layout_stride, dev_acc_d> s(buf.data(),
                                                                     map);

#if defined(GCXX_WITH_EXCEPTIONS)
  EXPECT_THROW(infer(s), gcxx::blas::BlasException);
#else
  EXPECT_DEATH(infer(s), "unit stride on one axis");
#endif
}

// Output-orientation inference: keys the transposed-output dispatch.
TEST(BlasOutputInference, ColumnMajorOutput_IsNotTransposed) {
  std::vector<double> buf(3 * 4);
  dmat2d<gcxx::layout_left> c(buf.data(), 3, 4);

  const auto out = gcxx::blas::details_::infer_blas_output_view(c);
  EXPECT_FALSE(out.transposed);
  EXPECT_EQ(out.rows, 3);
  EXPECT_EQ(out.cols, 4);
  EXPECT_EQ(out.leading_dimension, 3);
}

TEST(BlasOutputInference, RowMajorOutput_IsTransposed) {
  std::vector<double> buf(3 * 4);
  dmat2d<gcxx::layout_right> c(buf.data(), 3, 4);

  const auto out = gcxx::blas::details_::infer_blas_output_view(c);
  EXPECT_TRUE(out.transposed);
  EXPECT_EQ(out.rows, 3);
  EXPECT_EQ(out.cols, 4);
  EXPECT_EQ(out.leading_dimension, 4);
}

TEST(BlasOutputInference, NeitherAxisUnit_Rejected) {
  std::vector<double> buf(100);
  dextents2d ext{4, 5};
  gcxx::layout_stride::mapping<dextents2d> map{ext, std::array<int, 2>{2, 7}};
  gcxx::mdspan<double, dextents2d, gcxx::layout_stride, dev_acc_d> s(buf.data(),
                                                                     map);

#if defined(GCXX_WITH_EXCEPTIONS)
  EXPECT_THROW(gcxx::blas::details_::infer_blas_output_view(s),
               gcxx::blas::BlasException);
#else
  EXPECT_DEATH(gcxx::blas::details_::infer_blas_output_view(s),
               "unit stride on one axis");
#endif
}

// device-view gating traits; every check below is compile-time.
TEST(BlasDeviceView, AccessorTraitPolarity) {
  static_assert(gcxx::is_device_view_v<gcxx::device_accessor<def_acc_d>>,
                "device accessor is a device view");
  static_assert(gcxx::is_device_view_v<gcxx::managed_accessor<def_acc_d>>,
                "managed accessor is a device view");
  static_assert(gcxx::is_device_view_v<
                  gcxx::restrict_accessor<gcxx::device_accessor<def_acc_d>>>,
                "wrapper accessors propagate device-ness");
  static_assert(!gcxx::is_device_view_v<def_acc_d>,
                "default accessor is not a device view");
  static_assert(!gcxx::is_device_view_v<gcxx::host_accessor<def_acc_d>>,
                "host accessor is not a device view");
  static_assert(gcxx::is_host_view_v<gcxx::host_accessor<def_acc_d>>,
                "host accessor is a host view");
  static_assert(gcxx::is_host_view_v<gcxx::managed_accessor<def_acc_d>>,
                "managed accessor is a host view");
  static_assert(
    gcxx::is_restrict_accessor_v<gcxx::restrict_accessor<def_acc_d>>,
    "restrict accessor identity trait");
  static_assert(
    gcxx::is_shared_memory_accessor_v<gcxx::shared_memory_accessor<def_acc_d>>,
    "shared memory accessor identity trait");
  static_assert(
    !gcxx::is_device_view_v<gcxx::shared_memory_accessor<def_acc_d>>,
    "shared memory is not a BLAS device view (kernel-only "
    "address space)");
  SUCCEED();
}

TEST(BlasDeviceView, TransposedPreservesDeviceAccessor) {
  std::vector<double> buf(3 * 4);
  dmat2d<gcxx::layout_left> a(buf.data(), 3, 4);

  auto at = gcxx::transposed(a);
  static_assert(
    std::is_same_v<decltype(at)::accessor_type, dev_acc_d>,
    "transposed() forwards the device accessor, keeping the view BLAS-legal");
  SUCCEED();
}

TEST(BlasDeviceView, CrossSpaceConversionsAreDeleted) {
  using host_acc = gcxx::host_accessor<def_acc_d>;
  using dev_acc  = gcxx::device_accessor<def_acc_d>;
  using man_acc  = gcxx::managed_accessor<def_acc_d>;

  static_assert(!std::is_constructible_v<dev_acc, host_acc>,
                "device <- host accessor conversion is deleted");
  static_assert(!std::is_constructible_v<host_acc, dev_acc>,
                "host <- device accessor conversion is deleted");
  static_assert(!std::is_constructible_v<man_acc, host_acc>,
                "managed <- host accessor conversion is deleted");
  static_assert(!std::is_constructible_v<man_acc, dev_acc>,
                "managed <- device accessor conversion is deleted");
  static_assert(std::is_constructible_v<dev_acc, man_acc>,
                "device <- managed accessor conversion is allowed");
  static_assert(std::is_constructible_v<host_acc, man_acc>,
                "host <- managed accessor conversion is allowed");
  static_assert(std::is_constructible_v<dev_acc, def_acc_d>,
                "device <- plain accessor conversion is allowed");
  SUCCEED();
}
