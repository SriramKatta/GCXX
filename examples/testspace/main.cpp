#include <fmt/format.h>
#include <algorithm>
#include <gcxx/api.hpp>
#include <type_traits>

GCXX_TEMPLATE(typename VT)
GCXX_REQUIRES(std::is_arithmetic_v<VT> GCXX_AND std::is_floating_point_v<VT> ||
              std::is_integral_v<VT>)

void print_float_int(const VT& val) {
  fmt::print("{}\n", val);
}

template <typename VT>
auto reciprocal(const VT& value) GCXX_TRAILING_REQUIRES(VT)(
  std::is_floating_point_v<VT>&& std::is_arithmetic_v<VT>) {
  return VT{1} / value;
}

int main() {
  print_float_int(3);
  print_float_int(3.2);
  reciprocal(3.2);
  // reciprocal(3);
}
