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
    if (h_a[i] != checkval) {
      std::cout << "FAILED" << h_a[i] << std::endl;
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

  size_t size = N * sizeof(double);

  double* h_a = nullptr;
  double* d_a = nullptr;

  cudaMallocHost(&h_a, size);
  cudaMalloc(&d_a, size);

  std::memset(h_a, 0, size);

  gcxx::Event H2Dstart, H2Dend, D2Hstart, D2Hend, kernelstart, kernelend;

  H2Dstart.RecordInStream();
  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  H2Dend.RecordInStream();

  cudaDeviceSynchronize();

  kernelstart.RecordInStream();
  for (size_t i = 1; i <= rep; i++) {
    kernel_4vec<<<blocks, threads>>>(N, d_a);
  }
  kernelend.RecordInStream();

  cudaDeviceSynchronize();

  D2Hstart.RecordInStream();
  cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost);
  D2Hend.RecordInStream();

  checkdata(N, h_a, rep);

  float Dtohtime =
    (D2Hend.ElapsedTimeSince<gcxx::details_::secDuration_t>(D2Hstart)).count();
  float kerneltime =
    (kernelend.ElapsedTimeSince<gcxx::details_::secDuration_t>(kernelstart))
      .count();
  float HtoDtime =
    (H2Dend.ElapsedTimeSince<gcxx::details_::secDuration_t>(H2Dstart)).count();

  double arraydatasizeinGbytes = static_cast<double>(N * sizeof(double)) / 1e9;

  std::cout << kerneltime << " "
            << (arraydatasizeinGbytes * 2 * rep) / kerneltime << std::endl
            << Dtohtime << " " << arraydatasizeinGbytes / Dtohtime << std::endl
            << HtoDtime << " " << arraydatasizeinGbytes / HtoDtime << std::endl;
  cudaFreeHost(h_a);
  cudaFree(d_a);
  return 0;
}