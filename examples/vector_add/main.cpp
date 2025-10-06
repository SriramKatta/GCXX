#include <fmt/format.h>
#include <gcxx/api.hpp>

#include "main.hpp"

void checkdata(const gcxx::span<double>& h_a, double checkval) {
  for (size_t i = 0; i < h_a.size(); i++) {
    if ((h_a[i] - checkval) > 0.00001) {
      fmt::print("FAILED at index {} : {}\n", i, h_a[i] - checkval);
      exit(1);
    }
  }
}

template <typename func_t>
float time_measure(const gcxx::Stream& str, const Args& arg,
                   gcxx::span<double>& d_a_span, func_t func) {
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
  // using namespace gcxx::details_;

  Args arg = parse_args(argc, argv);

  size_t sizeInBytes = arg.N * sizeof(double);

  double* h_a{nullptr};
  double* d_a{nullptr};

#if GCXX_HIP_MODE
  GCXX_SAFE_RUNTIME_CALL(HostMalloc, "failed to allocated Pinned Host data",
                         &h_a, sizeInBytes);
#elif GCXX_CUDA_MODE
  GCXX_SAFE_RUNTIME_CALL(MallocHost, "failed to allocated Pinned Host data",
                         &h_a, sizeInBytes);
#endif


  GCXX_SAFE_RUNTIME_CALL(Malloc, "Failed to allocted GPU memory", &d_a,
                         sizeInBytes);


  gcxx::span h_a_span(h_a, arg.N);
  gcxx::span d_a_span(d_a, arg.N);


  std::memset(h_a_span.data(), 0, h_a_span.size_bytes());

  gcxx::Stream str(gcxx::flags::streamType::nullStream);

  auto H2Dstart = str.recordEvent();
  gcxx::memory::copy(d_a_span, h_a_span, str);
  auto H2Dend = str.recordEvent();

  auto scalar_kern_time =
    time_measure(str, arg, d_a_span, launch_scalar_kernel);
  auto vec2_kern_time = time_measure(str, arg, d_a_span, launch_vec2_kernel);
  auto vec4_kern_time = time_measure(str, arg, d_a_span, launch_vec4_kernel);


  auto D2Hstart = str.recordEvent();
  gcxx::memory::copy(h_a_span, d_a_span, str);
  auto D2Hend = str.recordEvent();

  D2Hend.Synchronize();

  checkdata(h_a_span, static_cast<double>(arg.rep) * 3.0);

  float Dtohtime = (D2Hend.ElapsedTimeSince<gcxx::sec>(D2Hstart)).count();

  float HtoDtime = (H2Dend.ElapsedTimeSince<gcxx::sec>(H2Dstart)).count();

  double arraySizeinGbytes = static_cast<double>(arg.N * sizeof(double)) / 1e9;
  double transfer_size = arraySizeinGbytes * 2.0 * static_cast<double>(arg.rep);

  fmt::print(
    "{:>4.3f} {:>4.3f}\n{:>4.3f} {:>4.3f}\n{:>4.3f} {:>4.3f}\n{:>4.3f} "
    "{:>4.3f}\n{:>4.3f} {:>4.3f}\n",
    scalar_kern_time, transfer_size / scalar_kern_time, vec2_kern_time,
    transfer_size / vec2_kern_time, vec4_kern_time,
    transfer_size / vec4_kern_time, Dtohtime, arraySizeinGbytes / Dtohtime,
    HtoDtime, arraySizeinGbytes / HtoDtime);

  GCXX_SAFE_RUNTIME_CALL(FreeHost, "Failed to free Allocated Host data", h_a);
  GCXX_SAFE_RUNTIME_CALL(Free, "Failed to free Allocated GPU data", d_a);
  return 0;
}