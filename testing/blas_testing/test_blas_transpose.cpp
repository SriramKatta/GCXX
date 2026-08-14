// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Host-only tests for the BLAS view helpers: transpose() and the stride-based
// op/leading-dim inference (details/op_inference.hpp). These run without a GPU
// — they exercise pure mdspan-metadata logic.

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

  // The device-view counterpart the BLAS inference helpers require. The
  // inference tests are metadata-only, so the underlying pointer never needs
  // to be real device memory here.
  template <class Layout>
  using dmat2d = gcxx::mdspan<double, dextents2d, Layout, dev_acc_d>;

  // Infer the (op, ld) descriptor for a view — under test via the details
  // helper.
  template <class V>
  auto infer(const V& v) {
    return gcxx::blas::details_::infer_blas_matrix_view(v);
  }

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// transpose(): extents and strides swap, round-trips.
// ─────────────────────────────────────────────────────────────────────────────

TEST(BlasTranspose, SwapsExtentsAndStrides_LayoutLeft) {
  std::vector<double> buf(3 * 4);
  mat2d<gcxx::layout_left> a(buf.data(), 3, 4);  // strides (1, 3)

  auto at = gcxx::blas::transpose(a);
  EXPECT_EQ(at.extent(0), 4);
  EXPECT_EQ(at.extent(1), 3);
  EXPECT_EQ(at.mapping().stride(0), 3);
  EXPECT_EQ(at.mapping().stride(1), 1);
}

TEST(BlasTranspose, SwapsExtentsAndStrides_LayoutRight) {
  std::vector<double> buf(3 * 4);
  mat2d<gcxx::layout_right> a(buf.data(), 3, 4);  // strides (4, 1)

  auto at = gcxx::blas::transpose(a);
  EXPECT_EQ(at.extent(0), 4);
  EXPECT_EQ(at.extent(1), 3);
  EXPECT_EQ(at.mapping().stride(0), 1);
  EXPECT_EQ(at.mapping().stride(1), 4);
}

TEST(BlasTranspose, RoundTrips_LayoutLeft) {
  std::vector<double> buf(3 * 4);
  mat2d<gcxx::layout_left> a(buf.data(), 3, 4);

  auto att = gcxx::blas::transpose(gcxx::blas::transpose(a));
  EXPECT_EQ(att.extent(0), a.extent(0));
  EXPECT_EQ(att.extent(1), a.extent(1));
  EXPECT_EQ(att.mapping().stride(0), a.mapping().stride(0));
  EXPECT_EQ(att.mapping().stride(1), a.mapping().stride(1));
}

// Static extents must be preserved at the type level (extents<int,3,4> ->
// extents<int,4,3>), the layout must flip, and transpose(transpose(v)) must
// restore the exact original type. This is the property the dextents-only
// version could not provide.
TEST(BlasTranspose, PreservesStaticExtentsAndType) {
  double buf[12] = {};
  gcxx::mdspan<double, gcxx::extents<int, 3, 4>, gcxx::layout_left,
               gcxx::default_accessor<double>>
    a(buf);

  auto at = gcxx::blas::transpose(a);
  static_assert(
    std::is_same_v<decltype(at)::extents_type, gcxx::extents<int, 4, 3>>,
    "static extents transpose 3x4 -> 4x3");
  static_assert(std::is_same_v<decltype(at)::layout_type, gcxx::layout_right>,
                "layout_left transposes to layout_right");
  EXPECT_EQ(at.extent(0), 4);
  EXPECT_EQ(at.extent(1), 3);

  auto att = gcxx::blas::transpose(at);
  static_assert(
    std::is_same_v<decltype(att)::extents_type, gcxx::extents<int, 3, 4>>,
    "round-trip restores original static extents");
  static_assert(std::is_same_v<decltype(att)::layout_type, gcxx::layout_left>,
                "round-trip restores original layout");
}

// ─────────────────────────────────────────────────────────────────────────────
// op / leading-dim inference: op depends only on which axis is unit-stride;
// dims are always the view's extents.
// ─────────────────────────────────────────────────────────────────────────────

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

  const auto info =
    infer(gcxx::blas::transpose(a));  // extents (4,3), strides (3,1)
  EXPECT_EQ(info.op, gcxx::driver::deviceBlasOpT);
  EXPECT_EQ(info.rows, 4);
  EXPECT_EQ(info.cols, 3);
  EXPECT_EQ(info.leading_dimension, 3);
}

TEST(BlasOpInference, TransposedLayoutRight_IsOpN) {
  std::vector<double> buf(3 * 4);
  dmat2d<gcxx::layout_right> a(buf.data(), 3, 4);

  const auto info =
    infer(gcxx::blas::transpose(a));  // extents (4,3), strides (1,4)
  EXPECT_EQ(info.op, gcxx::driver::deviceBlasOpN);
  EXPECT_EQ(info.rows, 4);
  EXPECT_EQ(info.cols, 3);
  EXPECT_EQ(info.leading_dimension, 4);
}

// A 1xN matrix has strides (1,1) — both unit. The tie-break must yield op=N
// (not reject it).
TEST(BlasOpInference, DegenerateUnitStrides_TieBreaksToOpN) {
  std::vector<double> buf(4);
  dmat2d<gcxx::layout_left> a(buf.data(), 1, 4);  // strides (1, 1)

  const auto info = infer(a);
  EXPECT_EQ(info.op, gcxx::driver::deviceBlasOpN);
  EXPECT_EQ(info.rows, 1);
  EXPECT_EQ(info.cols, 4);
  EXPECT_EQ(info.leading_dimension, 1);
}

// A genuinely strided subview (no unit stride on either axis) is rejected.
TEST(BlasOpInference, NeitherAxisUnit_Throws) {
  std::vector<double> buf(100);
  dextents2d ext{4, 5};
  gcxx::layout_stride::mapping<dextents2d> map{ext, std::array<int, 2>{2, 7}};
  gcxx::mdspan<double, dextents2d, gcxx::layout_stride, dev_acc_d> s(
    buf.data(), map);

  EXPECT_THROW(infer(s), gcxx::blas::BlasException);
}

// ─────────────────────────────────────────────────────────────────────────────
// Device-view gating (host_device_accessor.hpp): trait polarity, transpose
// preservation, and the deleted cross-memory-space conversions. All
// compile-time checks.
// ─────────────────────────────────────────────────────────────────────────────

TEST(BlasDeviceView, AccessorTraitPolarity) {
  static_assert(gcxx::is_device_view_v<gcxx::device_accessor<def_acc_d>>,
                "device accessor is a device view");
  static_assert(gcxx::is_device_view_v<gcxx::managed_accessor<def_acc_d>>,
                "managed accessor is a device view");
  static_assert(gcxx::is_device_view_v<gcxx::restrict_accessor<
                  gcxx::device_accessor<def_acc_d>>>,
                "wrapper accessors propagate device-ness");
  static_assert(!gcxx::is_device_view_v<def_acc_d>,
                "default accessor is not a device view");
  static_assert(!gcxx::is_device_view_v<gcxx::host_accessor<def_acc_d>>,
                "host accessor is not a device view");
  static_assert(gcxx::is_host_view_v<gcxx::host_accessor<def_acc_d>>,
                "host accessor is a host view");
  static_assert(gcxx::is_host_view_v<gcxx::managed_accessor<def_acc_d>>,
                "managed accessor is a host view");
  static_assert(gcxx::is_restrict_accessor_v<
                  gcxx::restrict_accessor<def_acc_d>>,
                "restrict accessor identity trait");
  static_assert(gcxx::is_shared_memory_accessor_v<
                  gcxx::shared_memory_accessor<def_acc_d>>,
                "shared memory accessor identity trait");
  static_assert(!gcxx::is_device_view_v<gcxx::shared_memory_accessor<def_acc_d>>,
                "shared memory is not a BLAS device view (kernel-only "
                "address space)");
  SUCCEED();
}

TEST(BlasDeviceView, TransposePreservesDeviceAccessor) {
  std::vector<double> buf(3 * 4);
  dmat2d<gcxx::layout_left> a(buf.data(), 3, 4);

  auto at = gcxx::blas::transpose(a);
  static_assert(
    std::is_same_v<decltype(at)::accessor_type, dev_acc_d>,
    "transpose() forwards the device accessor, keeping the view BLAS-legal");
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
