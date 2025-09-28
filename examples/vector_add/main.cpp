#include <fmt/format.h>
#include <gcxx/api.hpp>

#include "main.hpp"

void checkdata(size_t N, double* h_a, double checkval) {
  for (size_t i = 0; i < N; i++) {
    if ((h_a[i] - checkval) > 0.00001) {
      fmt::print("FAILED at index {} : {}\n", i, h_a[i]);
      exit(1);
    }
  }
}

int main(int argc, char** argv) {
  using namespace gcxx::details_;

  Args arg = parse_args(argc, argv);

  size_t sizeInBytes = arg.N * sizeof(double);

  double* h_a = nullptr;
  double* d_a = nullptr;

#if GCXX_HIP_MODE
  GCXX_SAFE_RUNTIME_CALL(HostMalloc, (&h_a, sizeInBytes));
#elif GCXX_CUDA_MODE
  GCXX_SAFE_RUNTIME_CALL(MallocHost, (&h_a, sizeInBytes));
#endif


  GCXX_SAFE_RUNTIME_CALL(Malloc, (&d_a, sizeInBytes));

  std::memset(h_a, 0, sizeInBytes);

  gcxx::Stream str(gcxx::flags::streamType::syncWithNull);

  auto H2Dstart = str.recordEvent();
  gcxx::memory::copy(d_a, h_a, arg.N, str);
  auto H2Dend = str.recordEvent();

  GCXX_SAFE_RUNTIME_CALL(DeviceSynchronize, ());

  str.Synchronize();
  auto kernelstart = str.recordEvent();
  for (size_t i = 1; i <= arg.rep; i++) {
    launch_vec4_kernel(arg, str, arg.N, d_a);
  }
  auto kernelend = str.recordEvent();

  str.Synchronize();
  // GCXX_SAFE_RUNTIME_CALL(DeviceSynchronize,());

  auto D2Hstart = str.recordEvent();
  gcxx::memory::copy(h_a, d_a, arg.N, str);
  auto D2Hend = str.recordEvent();

  str.Synchronize();

  checkdata(arg.N, h_a, static_cast<double>(arg.N));

  float Dtohtime   = (D2Hend.ElapsedTimeSince<sec>(D2Hstart)).count();
  float kerneltime = (kernelend.ElapsedTimeSince<sec>(kernelstart)).count();
  float HtoDtime   = (H2Dend.ElapsedTimeSince<sec>(H2Dstart)).count();

  double arraySizeinGbytes = static_cast<double>(arg.N * sizeof(double)) / 1e9;
  double transfer_size = arraySizeinGbytes * 2.0 * static_cast<double>(arg.rep);

  fmt::print("{} {}\n{} {}\n{} {}\n", kerneltime, transfer_size / kerneltime,
             Dtohtime, arraySizeinGbytes / Dtohtime, HtoDtime,
             arraySizeinGbytes / HtoDtime);

  GCXX_SAFE_RUNTIME_CALL(FreeHost, (h_a));
  GCXX_SAFE_RUNTIME_CALL(Free, (d_a));
  return 0;
}