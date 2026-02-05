#include <fmt/chrono.h>  // needed to print the chrono durations
#include <fmt/format.h>
#include <array>
#include <gcxx/api.hpp>
#include <numeric>

constexpr int nx = 5;
constexpr int ny = 5;

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
  gcxx::mdspan<int, gcxx::dextents<int, 2>, gcxx::layout_left,
               gcxx::restrict_accessor<gcxx::default_accessor<int>>>
    span(vec.data(), nx, ny);
  print(span);
  return 0;
}
