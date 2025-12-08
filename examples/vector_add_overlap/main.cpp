#include <vector>

#include <fmt/format.h>
#include <gcxx/api.hpp>

#include "main.hpp"

using datatype = int;

template <typename VT>
void checkdata(const gcxx::span<VT>& h_a, VT checkval) {
  for (size_t i = 0; i < h_a.size(); i++) {
    if ((h_a[i] - checkval) > 0.00001) {
      fmt::print("FAILED at index {} : {}\n", i, h_a[i] - checkval);
      exit(1);
    }
  }
  fmt::print("ALL PASSED!\n");
}

template <typename VT, typename func_t>
float time_measure(const gcxx::Stream& str, const Args& arg,
                   gcxx::span<VT>& d_a_span, func_t func) {
  str.Synchronize();
  auto kernelstart = str.recordEvent();
  for (size_t i = 1; i <= arg.rep; i++) {
    func(arg, str, d_a_span);
  }
  auto kernelend = str.recordEvent();
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


  std::memset(h_a_span.data(), 0, h_a_span.size_bytes());

  std::vector<gcxx::Stream> streams;
  streams.reserve(arg.numstreams);
  for (size_t i =0; i < arg.numstreams; ++i) {
    streams.emplace_back(gcxx::flags::streamType::syncWithNull);
  }

  for (size_t rep = 0; rep < arg.rep; rep++) {
    size_t base_count = arg.N / arg.numstreams;
    size_t i          = 0;
    for (auto& stream : streams) {
      size_t offset  = i++ * base_count;
      size_t count   = std::min(base_count, arg.N - 1 - offset);
      auto h_subspan = h_a_span.subspan(offset, count);
      auto d_subspan = d_a_span.subspan(offset, count);
      gcxx::memory::copy(d_subspan, h_subspan, stream);
      launch_scalar_kernel(arg, stream, d_subspan);
      launch_vec2_kernel(arg, stream, d_subspan);
      launch_vec4_kernel(arg, stream, d_subspan);
      gcxx::memory::copy(h_subspan, d_subspan, stream);
    }
  }

  GCXX_SAFE_RUNTIME_CALL(DeviceSynchronize, "FAILED to synchronize the device");

  checkdata(h_a_span, static_cast<datatype>(3 * arg.rep));

  return 0;
}