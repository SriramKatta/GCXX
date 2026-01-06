#include <fmt/format.h>
#include <algorithm>
#include <gcxx/api.hpp>

#include "main.hpp"

constexpr float keps = 1e-6;
using datatype       = float;

template <typename VT, typename func_t>
float time_measure(const gcxx::Stream& str, const Args& arg,
                   gcxx::span<VT>& d_a_span, func_t func) {
  str.Synchronize();
  auto kernelstart = str.RecordEvent();
  for (size_t i = 1; i <= arg.rep; i++) {
    func(arg, str, d_a_span);
  }
  auto kernelend = str.RecordEvent();
  str.Synchronize();
  float kerneltime =
    (kernelend.ElapsedTimeSince<gcxx::sec>(kernelstart)).count();
  return kerneltime;
}

int main(int argc, char** argv) {

  Args arg = parse_args(argc, argv);

  auto h_a = gcxx::host_vector<datatype>(arg.N);
  auto d_a = gcxx::device_vector<datatype>(arg.N);

  gcxx::span h_a_span(h_a);
  gcxx::span d_a_span(d_a);


  std::fill(h_a.begin(), h_a.end(), 1.0);

  gcxx::Stream str(gcxx::flags::streamType::NoSyncWithNull);

  auto H2Dstart = str.RecordEvent();
  gcxx::memory::Copy(d_a_span, h_a_span, str);
  auto H2Dend = str.RecordEvent();

  auto res = launch_reduction_kernel<datatype>(arg, str, d_a_span);
  str.Synchronize();

  if ((res - static_cast<datatype>(arg.N) > keps)) {
    fmt::print("CHECK FAILED res {} and check val{}\n", res, arg.N);

  } else {
    fmt::print("CHECK PASSED\n");
  }

  return 0;
}