// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#include <fmt/chrono.h>  // needed to print the chrono durations
#include <fmt/format.h>
#include <array>
#include <gcxx/runtime/memory/spans/spans.hpp>
#include <numeric>

constexpr int nx = 5;
constexpr int ny = 5;

template <typename VT, std::size_t N>
void print_linear(const gcxx::span<VT, N>& sps) {
  for (auto& i : sps) {
    fmt::print("{:3d} ", i);
  }
  fmt::print("\n");
}

template <typename VT, typename EXT, typename LAY, typename acc>
void print(const gcxx::mdspan<VT, EXT, LAY, acc>& span) {
  for (size_t i = 0; i < span.extent(0); i++) {
    for (size_t j = 0; j < span.extent(1); j++) {
      fmt::print("{:3d} ", span(i, j));
    }
    fmt::print("\n");
  }
}

int main(int argc, char const* argv[]) {
  std::array<int, nx * ny> vec;
  std::iota(vec.begin(), vec.end(), 1);

  gcxx::span lin_span(vec);

  fmt::print("linear data : \n");
  print_linear(lin_span);

  gcxx::mdspan<int, gcxx::dextents<int, 2>, gcxx::layout_left,
               gcxx::restrict_accessor<gcxx::default_accessor<int>>>
    span(vec.data(), nx, ny);

  fmt::print("\nmdspan layout left data : \n");
  print(span);

  gcxx::mdspan<int, gcxx::dextents<int, 2>, gcxx::layout_right,
               gcxx::default_accessor<int>>
    span2(vec.data(), nx, ny);

  fmt::print("\nmdspan layout right data : \n");
  print(span2);
  return 0;
}
