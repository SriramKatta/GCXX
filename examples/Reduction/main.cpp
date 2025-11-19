#include <fmt/format.h>
#include <algorithm>
#include <gcxx/api.hpp>

#include "main.hpp"

using datatype = float;

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
  // using namespace gcxx::details_;

  Args arg = parse_args(argc, argv);

  size_t sizeInBytes = arg.N * sizeof(datatype);

  datatype* h_a{nullptr};
  datatype* d_a{nullptr};

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


  std::fill(h_a, h_a + arg.N, 1.0);

  gcxx::Stream str(gcxx::flags::streamType::noSyncWithNull);

  auto H2Dstart = str.recordEvent();
  gcxx::memory::copy(d_a_span, h_a_span, str);
  auto H2Dend = str.recordEvent();

  auto res = launch_reduction_kernel<datatype>(arg, str, d_a_span);

  if (res != arg.N) {
    fmt::print("CHECK FAILED res {} and check val{}\n", res, arg.N);

  } else {
    fmt::print("CHECK PASSED\n");
  }


  GCXX_SAFE_RUNTIME_CALL(FreeHost, "Failed to free Allocated Host data", h_a);
  GCXX_SAFE_RUNTIME_CALL(Free, "Failed to free Allocated GPU data", d_a);
  return 0;
}