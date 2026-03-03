#include <fmt/format.h>
#include <algorithm>
#include <gcxx/api.hpp>
#include <type_traits>

GCXX_TEMPLATE(class VT)
GCXX_REQUIRES((std::is_arithmetic_v<VT>) GCXX_AND (std::is_floating_point_v<VT> ||
               std::is_integral_v<VT>))

void print_float_int(const VT& val) {
  std::cout << val << std::endl;
}

int main() {
  print_float_int(3);
  print_float_int(3.2);
  // print_float_int('c');
}
