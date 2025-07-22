#include <fmt/color.h>
#include <gpucxx/api.hpp>
#include <vector>

constexpr float giga  = 1e9;
constexpr float milli = 1e3;

int main(){
  auto ev = gpuCXX::EventCreate();
}

int main0(int argc, char **argv) {
  if (argc == 1) {
    argv[1] = "10";
  }

  auto start_ev      = gpuCXX::EventCreate();
  auto stop_ev       = gpuCXX::EventCreate();
  int *h_a           = nullptr;
  int *d_a           = nullptr;
  size_t count       = atoll(argv[1]);
  size_t N           = 1 << count;
  size_t sizeInBytes = N * sizeof(int);
  cudaMallocHost(&h_a, sizeInBytes);
  cudaMalloc(&d_a, sizeInBytes);

  start_ev.RecordInStream();
  cudaMemcpyAsync(d_a, h_a, sizeInBytes, cudaMemcpyDefault);
  stop_ev.RecordInStream();
  fmt::print("{}\n", stop_ev.query());

  float sizeinGB = static_cast<float>(sizeInBytes) / giga;
  float ms       = stop_ev.ElapsedTimeSince(start_ev);
  fmt::print("bw is {}\n", sizeinGB / (ms / milli));

  cudaFree(d_a);
  cudaFreeHost(h_a);
  return 0;
}