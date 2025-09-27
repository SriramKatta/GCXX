#include <fmt/format.h>
#include <gpucxx/api.hpp>

__global__ void kernel_scalar(size_t N, double* a) {
  int start  = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (size_t i = start; i < N; i += stride) {
    a[i] = a[i] + 1.0;
  }
}

__global__ void kernel_2vec(size_t N, double* a) {
  int start  = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (size_t i = start; i < N / 2; i += stride) {
    double2* a2 = reinterpret_cast<double2*>(a) + i;
    a2->x       = a2->x + 1.0;
    a2->y       = a2->y + 1.0;
  }
  if (N % 2 != 0 && start == 0) {
    a[N - 1] += 1.0;
  }
}

__global__ void kernel_4vec(size_t N, double* a) {
  int start  = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (size_t i = start; i < N / 4; i += stride) {
    double4* a4 = reinterpret_cast<double4*>(a) + i;
    a4->x       = a4->x + 1.0;
    a4->y       = a4->y + 1.0;
    a4->z       = a4->z + 1.0;
    a4->w       = a4->w + 1.0;
  }
  int remainder = N % 4;
  if (start < remainder) {
    a[N - remainder + start] += 1.0;
  }
}

void checkdata(size_t N, double* h_a, double checkval) {
  for (size_t i = 0; i < N; i++) {
    if ((h_a[i] - checkval) > 0.00001) {
      fmt::print("FAILED at index {} : {}\n", i, h_a[i]);
      exit(1);
    }
  }
}

int main(int argc, char const* argv[]) {
  using namespace gcxx::details_;

  if (argc != 5) {
    fmt::print(
      "the useage is\n {} <Num elements> <num kernel repetitions> <num blocks> "
      "<threads per block>\n",
      argv[0]);
    return 1;
  }
  size_t N       = std::atoi(argv[1]);
  size_t rep     = std::atoi(argv[2]);
  size_t blocks  = std::atoi(argv[3]);
  size_t threads = std::atoi(argv[4]);

  size_t sizeInBytes = N * sizeof(double);

  double* h_a = nullptr;
  double* d_a = nullptr;

#if GCXX_HIP_MODE
  GCXX_SAFE_RUNTIME_CALL(HostMalloc, (&h_a, sizeInBytes));
#elif GCXX_CUDA_MODE
  GCXX_SAFE_RUNTIME_CALL(MallocHost, (&h_a, sizeInBytes));
#endif


  GCXX_SAFE_RUNTIME_CALL(Malloc, (&d_a, sizeInBytes));

  std::memset(h_a, 0, sizeInBytes);

  gcxx::Stream str(gcxx::flags::streamType::nullStream);

  auto H2Dstart = str.recordEvent();
  gcxx::memory::copy(d_a, h_a, N);
  auto H2Dend = str.recordEvent();

  GCXX_SAFE_RUNTIME_CALL(DeviceSynchronize, ());

  str.Synchronize();
  auto kernelstart = str.recordEvent();
  for (size_t i = 1; i <= rep; i++) {
    kernel_4vec<<<blocks, threads, 0, str.get()>>>(N, d_a);
  }
  auto kernelend = str.recordEvent();

  str.Synchronize();
  // GCXX_SAFE_RUNTIME_CALL(DeviceSynchronize,());

  auto D2Hstart = str.recordEvent();
  gcxx::memory::copy(h_a, d_a, N);
  auto D2Hend = str.recordEvent();

  checkdata(N, h_a, static_cast<double>(rep));

  float Dtohtime   = (D2Hend.ElapsedTimeSince<sec>(D2Hstart)).count();
  float kerneltime = (kernelend.ElapsedTimeSince<sec>(kernelstart)).count();
  float HtoDtime   = (H2Dend.ElapsedTimeSince<sec>(H2Dstart)).count();

  double arraySizeinGbytes = static_cast<double>(N * sizeof(double)) / 1e9;
  double transfer_size     = arraySizeinGbytes * 2.0 * static_cast<double>(rep);

  fmt::print("{} {}\n{} {}\n{} {}\n", kerneltime, transfer_size / kerneltime,
             Dtohtime, arraySizeinGbytes / Dtohtime, HtoDtime,
             arraySizeinGbytes / HtoDtime);

  GCXX_SAFE_RUNTIME_CALL(FreeHost, (h_a));
  GCXX_SAFE_RUNTIME_CALL(Free, (d_a));
  return 0;
}