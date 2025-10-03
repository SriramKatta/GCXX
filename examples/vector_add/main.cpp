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

int main(int argc, char** argv) {
  //using namespace gcxx::details_;

  Args arg = parse_args(argc, argv);

  size_t sizeInBytes = arg.N * sizeof(double);

  double* h_a{nullptr};
  double* d_a{nullptr};

#if GCXX_HIP_MODE
  GCXX_SAFE_RUNTIME_CALL(HostMalloc, (&h_a, sizeInBytes));
#elif GCXX_CUDA_MODE
  GCXX_SAFE_RUNTIME_CALL(MallocHost, (&h_a, sizeInBytes));
#endif


  GCXX_SAFE_RUNTIME_CALL(Malloc, (&d_a, sizeInBytes));


  gcxx::span h_a_span(h_a, arg.N);
  gcxx::span d_a_span(d_a, arg.N);


  std::memset(h_a_span.data(), 0, h_a_span.size_bytes());

  gcxx::Stream str(gcxx::flags::streamType::nullStream);

  auto H2Dstart = str.recordEvent();
  gcxx::memory::copy(d_a_span, h_a_span, str);
  auto H2Dend = str.recordEvent();

  H2Dend.Synchronize();

  str.Synchronize();
  auto kernelstart = str.recordEvent();
  for (size_t i = 1; i <= arg.rep; i++) {
    launch_vec2_kernel(arg, str, d_a_span);
  }
  auto kernelend = str.recordEvent();

  str.Synchronize();
  // GCXX_SAFE_RUNTIME_CALL(DeviceSynchronize,());

  auto D2Hstart = str.recordEvent();
  gcxx::memory::copy(h_a_span, d_a_span, str);
  auto D2Hend = str.recordEvent();

  D2Hend.Synchronize();

  checkdata(h_a_span, static_cast<double>(arg.rep));

  float Dtohtime   = (D2Hend.ElapsedTimeSince<gcxx::sec>(D2Hstart)).count();
  float kerneltime = (kernelend.ElapsedTimeSince<gcxx::sec>(kernelstart)).count();
  float HtoDtime   = (H2Dend.ElapsedTimeSince<gcxx::sec>(H2Dstart)).count();

  double arraySizeinGbytes = static_cast<double>(arg.N * sizeof(double)) / 1e9;
  double transfer_size = arraySizeinGbytes * 2.0 * static_cast<double>(arg.rep);

  fmt::print("{} {}\n{} {}\n{} {}\n", kerneltime, transfer_size / kerneltime,
             Dtohtime, arraySizeinGbytes / Dtohtime, HtoDtime,
             arraySizeinGbytes / HtoDtime);

  GCXX_SAFE_RUNTIME_CALL(FreeHost, (h_a));
  GCXX_SAFE_RUNTIME_CALL(Free, (d_a));
  return 0;
}