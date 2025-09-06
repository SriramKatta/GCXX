#include <cstring>
#include <gpucxx/api.hpp>
#include <iostream>

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
      std::cout << "FAILED " << h_a[i] << std::endl;
      exit(1);
    }
  }
}

int main(int argc, char const* argv[]) {

  if (argc != 5) {
    std::cout << "the useage is\n"
              << argv[0]
              << " <Num elements> <num kernel repetitions> <num blocks> "
                 "<threads per block>"
              << std::endl;
    return 1;
  }
  size_t N       = std::atoi(argv[1]);
  size_t rep     = std::atoi(argv[2]);
  size_t blocks  = std::atoi(argv[3]);
  size_t threads = std::atoi(argv[4]);

  size_t sizeInBytes = N * sizeof(double);

  double* h_a = nullptr;
  double* d_a = nullptr;

#if GPUCXX_HIP_MODE
  GPUCXX_SAFE_RUNTIME_CALL(HostMalloc, (&h_a, sizeInBytes));
#elseif GPUCXX_CUDA_MODE
  GPUCXX_SAFE_RUNTIME_CALL(MallocHost, (&h_a, sizeInBytes));
#endif


  GPUCXX_SAFE_RUNTIME_CALL(Malloc, (&d_a, sizeInBytes));

  std::memset(h_a, 0, sizeInBytes);

  gcxx::Stream str(gcxx::flags::streamType::nullStream);

  auto H2Dstart = str.recordEvent();
  gcxx::memory::copy(d_a, h_a, N);
  auto H2Dend = str.recordEvent();

  GPUCXX_SAFE_RUNTIME_CALL(DeviceSynchronize, ());

  str.Synchronize();
  auto kernelstart = str.recordEvent();
  for (size_t i = 1; i <= rep; i++) {
    kernel_4vec<<<blocks, threads, 0, str.get()>>>(N, d_a);
  }
  auto kernelend = str.recordEvent();

  str.Synchronize();
  // GPUCXX_SAFE_RUNTIME_CALL(DeviceSynchronize,());

  auto D2Hstart = str.recordEvent();
  gcxx::memory::copy(h_a, d_a, N);
  auto D2Hend = str.recordEvent();

  checkdata(N, h_a, rep);

  float Dtohtime =
    (D2Hend.ElapsedTimeSince<gcxx::details_::sec>(D2Hstart)).count();
  float kerneltime =
    (kernelend.ElapsedTimeSince<gcxx::details_::sec>(kernelstart)).count();
  float HtoDtime =
    (H2Dend.ElapsedTimeSince<gcxx::details_::sec>(H2Dstart)).count();

  double arraydatasizeinGbytes = static_cast<double>(N * sizeof(double)) / 1e9;

  std::cout << kerneltime << " "
            << (arraydatasizeinGbytes * 2 * rep) / kerneltime << std::endl
            << Dtohtime << " " << arraydatasizeinGbytes / Dtohtime << std::endl
            << HtoDtime << " " << arraydatasizeinGbytes / HtoDtime << std::endl;
  GPUCXX_SAFE_RUNTIME_CALL(FreeHost, (h_a));
  GPUCXX_SAFE_RUNTIME_CALL(Free, (d_a));
  return 0;
}