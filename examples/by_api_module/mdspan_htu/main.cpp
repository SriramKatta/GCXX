// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#include <fmt/chrono.h>  // needed to print the chrono durations
#include <fmt/format.h>
#include <array>
#include <gcxx/runtime/memory/spans/spans.hpp>
#include <numeric>
#include <utility>

constexpr int nx = 5;
constexpr int ny = 5;
using datatype   = float;
using def_acc    = gcxx::default_accessor<datatype>;
using res_acc    = gcxx::restrict_accessor<def_acc>;

using dextents2d = gcxx::dextents<int, 2>;

template <typename Layout, typename Acc>
using span2d = gcxx::mdspan<datatype, dextents2d, Layout, Acc>;


template <typename VT, std::size_t N>
void print_linear(const gcxx::span<VT, N>& sps);

template <typename VT, typename EXT, typename LAY, typename acc>
void print(const gcxx::mdspan<VT, EXT, LAY, acc>& span);

int main(int argc, char const* argv[]) {

  std::array<datatype, nx * ny> vec;
  std::iota(vec.begin(), vec.end(), static_cast<datatype>(1));

  gcxx::span lin_span(vec);

  fmt::println("linear data : ");
  print_linear(lin_span);

  fmt::println("\nsizeof pointer {}: ", sizeof(vec.data()));
  fmt::println("\nsizeof static span {}: ", sizeof(lin_span));

  span2d<gcxx::layout_left, res_acc> span(vec.data(), nx, ny);

  fmt::println("\nmdspan layout left data | sizeof mdspan {}: ", sizeof(span));
  print(span);

  span2d<gcxx::layout_right, def_acc> span2(vec.data(), nx, ny);

  fmt::println("\nmdspan layout right data | sizeof mdspan {}: ",
               sizeof(span2));
  print(span2);

  gcxx::mdspan<datatype, gcxx::extents<size_t, gcxx::dynamic_extent, ny>,
               gcxx::layout_right, def_acc>
    span3(vec.data(), nx);

  fmt::println(
    "\nmdspan dynamic, static extents layout right data | sizeof mdspan {}: ",
    sizeof(span3));
  print(span3);


  auto subspan = gcxx::submdspan(span3, std::pair{1, 3}, std::pair{1, 3});
  fmt::println(
    "\nmdspan subspan of dynamic, static extents layout right data | sizeof "
    "mdspan {}: ",
    sizeof(subspan));
  print(subspan);

  return 0;
}

template <typename VT, std::size_t N>
void print_linear(const gcxx::span<VT, N>& sps) {
  for (auto& i : sps) {
    fmt::print("{:3} ", i);
  }
  fmt::println("");
}

template <typename VT, typename EXT, typename LAY, typename acc>
void print(const gcxx::mdspan<VT, EXT, LAY, acc>& span) {
  fmt::println("mdspan rank {} | extent(0) {} | extent(1) {} ", span.rank(),
               span.extent(0), span.extent(1));
  fmt::println("stride(0) {} | stride(1) {} ", span.mapping().stride(0),
               span.mapping().stride(1));
  for (size_t i = 0; i < span.extent(0); i++) {
    for (size_t j = 0; j < span.extent(1); j++) {
      fmt::print("{:3} ", span(i, j));
    }
    fmt::println("");
  }
}
